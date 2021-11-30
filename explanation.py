import numpy as np
import itertools
from tqdm import tqdm_notebook
import networkx as nx
import pandas as pd


class ClustersExplanation:
    thresholds = None
    clusters = None
    features = None
    min_length = None
    max_length = None
    must_have_edges = None
    
    thresholds_df = None
    full_edges = None
    comparison_edges = None
    comparison_graph = None
    comparison_edge_features_dictionary = None
    feature_rules = None
    bipartite_comparison_subgraphs = None
    subgraph_sets = None
    subgraph_set_params = None
    subgraph_set_meanings = None
    best_coverage = None
    
    interpreted_features = None
    
    def __init__(self, clusters, features, thresholds=np.linspace(0, 0.3, 26), min_length=3, max_length=9, must_have_edges=set()): 
        self.clusters = clusters
        self.features = features
        self.thresholds = thresholds
        self.min_length = min_length
        self.max_length = max_length
        self.must_have_edges = must_have_edges
    
    def fit(self, users_df):
        self._calculate_thresholds(users_df)
        self._generate_fully_connected_edges()
        self._generate_comparison_edges()
        self._create_comparison_graph()
        self._find_bipartite_comparison_subgraphs()
        self._generate_subgraph_sets()
        self._find_best_comparison_graph_coverage()
    
    def _calculate_thresholds(self, users_df):
        features = self.features
        all_thresholds = self.thresholds
        
        self.thresholds_df = users_df.groupby("cluster")[features].quantile(
            list(set(np.hstack([all_thresholds, 1 - all_thresholds])))
        )
        
    def _generate_fully_connected_edges(self):
        clusters = self.clusters
        features = self.features
        
        full_edges = []
        
        for feature in tqdm_notebook(features):
            for A in clusters:
                for B in clusters:
                    if A == B:
                        continue
                    full_edges.append((A, B))
                    
        self.full_edges = full_edges

    def _generate_comparison_edges(self):
        clusters = self.clusters
        features = self.features
        all_thresholds = self.thresholds
        thresholds_df = self.thresholds_df
                
        edges = []

        for feature in features:
            for A in clusters:
                for B in clusters:
                    if A == B:
                        continue

                    selected_threshold = None

                    for threshold in all_thresholds:
                        A_upper_threshold = thresholds_df.loc[(A, 1 - threshold), feature]
                        B_lower_threshold = thresholds_df.loc[(B, threshold), feature]
                        if A_upper_threshold < B_lower_threshold:
                            selected_threshold = threshold
                            break

                    if not selected_threshold:
                        continue

                    edges.append((A, B, {
                        "feature": feature, 
                        "threshold": selected_threshold, 
                        "upper": A_upper_threshold, 
                        "lower": B_lower_threshold
                    }))
        
        self.comparison_edges = edges
    
    def _create_comparison_graph(self):
        graph = nx.DiGraph()
        edges = self.comparison_edges
        
        graph.add_edges_from(edges)

        edge_features = {}

        for A, B, d in edges:
            e = (A, B)
            if e not in edge_features:
                edge_features[e] = []
            edge_features[e].append(d["feature"])
        
        self.comparison_graph = graph
        self.comparison_edge_features_dictionary = edge_features
    
#     def _show_graph(self, graph, edge_features):
#         plt.figure(figsize=(10, 10))
#         pos = nx.shell_layout(graph)
#         nx.draw(graph, pos, with_labels = True, font_color='black')
#         nx.draw_networkx_edge_labels(graph, pos, edge_labels={a: ", ".join([b[0:20] for b in bb]) for a, bb in edge_features.items()}, font_color='red')
#         plt.show()
        
    def _get_non_oriented_edges(self, edges):
        return set([tuple(sorted(t)) for t in edges])
        
    def _find_bipartite_comparison_subgraphs(self):
        edges = self.comparison_edges
        features = self.features
        feature_rules = {}

        for selected_feature in features:
            # Find feature-related edges, check if the graph is not multi
            f_edges = [(A, B, d) for A, B, d in edges if d["feature"] == selected_feature]
            f_edges_only = [(A, B) for A, B, _ in f_edges]
            assert len(f_edges_only) == len(set(f_edges_only)), "Edges in both directions are not supported"

            # Build feature graph
            f_graph = nx.DiGraph()
            f_graph.add_edges_from(f_edges)

            # Build set of overlapping segments (with ending points)
            segments = [
                (e, f_graph.get_edge_data(*e)["upper"], f_graph.get_edge_data(*e)["lower"])
                for e in f_graph.edges
            ]
            segment_points = [(s[0], s[1], 1) for s in segments] + [(s[0], s[2], -1) for s in segments]

            # Find all segment intersections. 
            current_intersection_size = 0
            current_intersection = set()
            all_intersections = []
            previously_dropped = False

            for edge, point, beginning in sorted(segment_points, key=lambda x: x[1]):
                current_intersection_size += beginning
                if beginning > 0:
                    current_intersection.add(edge)
                    previously_dropped = False
                else:
                    if not previously_dropped:
                        all_intersections.append(current_intersection.copy())
                    previously_dropped = True
                    current_intersection.remove(edge)

            # Select independent splittable intersection groups
            rules = []

            for intersection in all_intersections:
                graph_df = pd.DataFrame([(
                    e[0],
                    e[1],
                    f_graph.get_edge_data(*e)["threshold"],
                    f_graph.get_edge_data(*e)["upper"],
                    f_graph.get_edge_data(*e)["lower"],
                ) for e in f_graph.edges if e in intersection], columns=["A", "B", "threshold", "upper", "lower"])
                part_A, part_B = graph_df["A"].unique().tolist(), graph_df["B"].unique().tolist()
                if len([a for a in part_A if a in part_B]) > 0:
                    continue
                # TODO check if this is a correct merge
                rule = graph_df.agg({
                    "threshold": "max",
                    "upper": "max",
                    "lower": "min"
                }).to_dict()
                rules.append(
                    (part_A, rule["threshold"], rule["upper"], rule["lower"], part_B, intersection)
                )
            feature_rules[selected_feature] = rules
        
        self.bipartite_comparison_subgraphs = feature_rules
    
    def _generate_subgraph_sets(self):
        feature_rules = self.bipartite_comparison_subgraphs
        subsets = []
        subset_params = []
        subset_meaning = []
        
        for feature, rules in feature_rules.items():
            for (part_A, threshold, upper, lower, part_B, subset) in rules:
                subsets.append(self._get_non_oriented_edges(subset))
                subset_params.append({
                    "probability": 1 - threshold,
                    "feature": feature,
                    "index": len(subset_params)
                })
                subset_meaning.append((feature, part_A, threshold, upper, lower, part_B, subset))

        self.subgraph_sets = subsets
        self.subgraph_set_params = subset_params
        self.subgraph_set_meanings = subset_meaning
    
    def _find_best_comparison_graph_coverage(self):
        min_length = self.min_length
        max_length = self.max_length
        must_have_edges = self.must_have_edges
        full_edges = self.full_edges
        
        universe = self._get_non_oriented_edges(full_edges)
        best_score = 100050000
        best_coverage = []
        
        subsets = self.subgraph_sets
        subset_params = self.subgraph_set_params

        for length in tqdm_notebook(range(min_length, max_length + 1)):
            for coverage in list(itertools.combinations(range(len(subsets)), length)):
                score = 1 #- np.median([subset_params[i]["probability"] for i in coverage])
                result = set([s for i in coverage for s in subsets[i]])
                score *= len(universe - result)
                features = [
                    subset_params[i]['feature']
                    for i in coverage
                ]
                if (len(features) == len(set(features))) and score < best_score and all([e in result for e in must_have_edges]):
                    best_score = score
                    best_coverage = coverage

                if best_score == 0:
                    break
        
        self.best_coverage = best_coverage
        self.interpreted_features = [self.subgraph_set_meanings[i] for i in self.best_coverage]
        
        return best_score
    
    def explain(self):
        return [
            dict(zip(("feature", "lower_part", "threshold", "value", "upper_part"), (f, a, t, u, b))) 
            for f, a, t, u, _, b, _ in self.interpreted_features
        ]
    
    def get_cluster_names(self, rule_interpretation, other_modifier="other"):
        clusters = self.clusters
        selected_features = self.interpreted_features
        
        cluster_names = {}

        for cluster in clusters:
            modifiers = []
            for i, (f, a, t, u, _, b, _) in enumerate(selected_features):
                if cluster in a:
                    modifiers.append(rule_interpretation[i][0])
                elif cluster in b:
                    modifiers.append(rule_interpretation[i][1])
            modifiers.append("users")
            modifiers = [m for m in modifiers if m]
            if len(modifiers) == 1:
                modifiers.insert(0, other_modifier)
            cluster_names[cluster] = " ".join(modifiers).capitalize()

        return cluster_names
    
    def get_legend(self, rule_interpretation):
        selected_features = self.interpreted_features
        for i, (f, a, t, u, _, b, _) in enumerate(selected_features):
            interpretation = rule_interpretation[i]
            if interpretation[0]:
                sign = "<"
            else:
                sign = ">"

            modifier = interpretation[0] + " / " + interpretation[1]
            print(modifier, ":", f, sign, "{:.3f}".format(u), "for", "{:.1f}% of cluster".format((1 - t) * 100))
    
    def score(self):
        full_edges_set = self._get_non_oriented_edges(self.full_edges)
        selected_edges = set([s for i in self.best_coverage for s in self.subgraph_sets[i]])
        return len(selected_edges) / len(full_edges_set)