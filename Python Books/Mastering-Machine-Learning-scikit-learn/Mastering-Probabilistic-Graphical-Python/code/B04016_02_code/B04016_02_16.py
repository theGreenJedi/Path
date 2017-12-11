# For creating a chordal graph using first heuristic there are six
# heuristics that are implemented in pgmpy and can be used by
# passing the keyword argument heuristic as H1, H2, H3, H4, H5, H6
chordal_graph = model.triangulate(heuristic='H1')
