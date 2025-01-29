import tensorflow as tf

# Load the protobuf file
graph_def = tf.compat.v1.GraphDef()

# Path to the file
file_path = "path_to_your_file/graph_opt.pb"

# Read the file
with tf.io.gfile.GFile(file_path, "rb") as f:
    graph_def.ParseFromString(f.read())

# Import the graph into a new Graph
with tf.Graph().as_default() as graph:
    tf.import_graph_def(graph_def, name="")

# List all operations
for op in graph.get_operations():
    print(op.name)

# Create a session and import the graph
with tf.compat.v1.Session() as sess:
    tf.import_graph_def(graph_def, name="")
    # Print the input and output tensor names
    for op in sess.graph.get_operations():
        print(op.name)
