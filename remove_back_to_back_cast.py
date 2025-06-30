#!/usr/bin/env python3
"""
Remove back-to-back Cast operations in ONNX models.

This script identifies and removes redundant Cast operations that occur consecutively:
1. Cast A -> Cast B where A and B have the same data type (removes both)
2. Cast A -> Cast B where they can be combined into a single cast

The script handles both the main graph and any subgraphs (e.g., in If, Loop, Scan nodes).
"""

import onnx
from onnx import TensorProto
import argparse
import sys
from typing import List, Tuple, Optional


def get_data_type_name(data_type: int) -> str:
    """Convert ONNX data type enum to readable string."""
    type_map = {
        TensorProto.FLOAT: "float32",
        TensorProto.UINT8: "uint8",
        TensorProto.INT8: "int8",
        TensorProto.UINT16: "uint16",
        TensorProto.INT16: "int16",
        TensorProto.INT32: "int32",
        TensorProto.INT64: "int64",
        TensorProto.STRING: "string",
        TensorProto.BOOL: "bool",
        TensorProto.FLOAT16: "float16",
        TensorProto.DOUBLE: "float64",
        TensorProto.UINT32: "uint32",
        TensorProto.UINT64: "uint64",
        TensorProto.COMPLEX64: "complex64",
        TensorProto.COMPLEX128: "complex128",
        TensorProto.BFLOAT16: "bfloat16",
    }
    return type_map.get(data_type, f"unknown({data_type})")


def find_back_to_back_casts(
    graph: onnx.GraphProto,
) -> List[Tuple[int, int, str, int, int]]:
    """
    Find back-to-back Cast operations in the graph.

    Returns:
        List of tuples: (first_cast_idx, second_cast_idx, connection_name, from_type, to_type)
    """
    cast_nodes = []
    node_by_output = {}

    # Find all Cast nodes and build output mapping
    for i, node in enumerate(graph.node):
        if node.op_type == "Cast":
            cast_nodes.append((i, node))
        for output in node.output:
            node_by_output[output] = (i, node)

    back_to_back_casts = []

    # Check each Cast node to see if its input comes from another Cast
    for cast_idx, cast_node in cast_nodes:
        input_name = cast_node.input[0]

        # Check if input comes from another Cast node
        if input_name in node_by_output:
            producer_idx, producer_node = node_by_output[input_name]
            if producer_node.op_type == "Cast":
                # Found back-to-back casts
                # Get data types
                from_type = None
                to_type_intermediate = None
                to_type_final = None

                for attr in producer_node.attribute:
                    if attr.name == "to":
                        to_type_intermediate = attr.i

                for attr in cast_node.attribute:
                    if attr.name == "to":
                        to_type_final = attr.i

                # Try to infer the original type from the graph inputs or previous nodes
                producer_input = producer_node.input[0]
                from_type = get_input_type(graph, producer_input)

                back_to_back_casts.append(
                    (
                        producer_idx,
                        cast_idx,
                        input_name,
                        from_type
                        or to_type_intermediate,  # fallback if we can't determine original type
                        to_type_final,
                    )
                )

    return back_to_back_casts


def get_input_type(graph: onnx.GraphProto, input_name: str) -> Optional[int]:
    """Get the data type of a graph input or initializer."""
    # Check graph inputs
    for input_info in graph.input:
        if input_info.name == input_name:
            return input_info.type.tensor_type.elem_type

    # Check initializers
    for init in graph.initializer:
        if init.name == input_name:
            return init.data_type

    return None


def remove_back_to_back_casts(
    model: onnx.ModelProto, verbose: bool = False
) -> onnx.ModelProto:
    """
    Remove back-to-back Cast operations from the model, including subgraphs.

    Args:
        model: Input ONNX model
        verbose: Print information about removed casts

    Returns:
        Optimized ONNX model
    """
    if verbose:
        print("Processing main graph...")

    modified = remove_back_to_back_casts_from_graph(model.graph, verbose)

    if not modified:
        if verbose:
            print("No back-to-back casts found in the entire model.")

    return model


def process_subgraphs(node: onnx.NodeProto, verbose: bool = False) -> bool:
    """
    Process subgraphs within a node (e.g., If, Loop, Scan operators).

    Args:
        node: ONNX node that might contain subgraphs
        verbose: Print information about processing

    Returns:
        True if any subgraphs were modified, False otherwise
    """
    modified = False

    for attr in node.attribute:
        if attr.type == onnx.AttributeProto.GRAPH:
            # Single subgraph (e.g., If node)
            if verbose:
                print(
                    f"    Processing subgraph in {node.op_type} node attribute '{attr.name}'"
                )

            subgraph_modified = remove_back_to_back_casts_from_graph(attr.g, verbose)
            if subgraph_modified:
                modified = True

        elif attr.type == onnx.AttributeProto.GRAPHS:
            # Multiple subgraphs (e.g., Loop node)
            for i, subgraph in enumerate(attr.graphs):
                if verbose:
                    print(
                        f"    Processing subgraph {i} in {node.op_type} node attribute '{attr.name}'"
                    )

                subgraph_modified = remove_back_to_back_casts_from_graph(
                    subgraph, verbose
                )
                if subgraph_modified:
                    modified = True

    return modified


def remove_back_to_back_casts_from_graph(
    graph: onnx.GraphProto, verbose: bool = False
) -> bool:
    """
    Remove back-to-back Cast operations from a specific graph.

    Args:
        graph: ONNX graph to process
        verbose: Print information about removed casts

    Returns:
        True if the graph was modified, False otherwise
    """
    # First, recursively process any subgraphs
    subgraph_modified = False
    for node in graph.node:
        if process_subgraphs(node, verbose):
            subgraph_modified = True

    # Then process this graph
    back_to_back_casts = find_back_to_back_casts(graph)

    if not back_to_back_casts:
        if verbose and not subgraph_modified:
            print("  No back-to-back casts found in this graph.")
        return subgraph_modified

    if verbose:
        print(
            f"  Found {len(back_to_back_casts)} back-to-back cast pairs in this graph:"
        )

    # Keep track of nodes to remove and connections to update
    nodes_to_remove = set()
    output_replacements = {}  # old_output -> new_output

    for first_idx, second_idx, _, from_type, to_type in back_to_back_casts:
        first_node = graph.node[first_idx]
        second_node = graph.node[second_idx]

        if verbose:
            print(
                f"    Cast {get_data_type_name(from_type)} -> {get_data_type_name(to_type)} "
                f"(removing intermediate cast to {get_data_type_name(from_type)})"
            )

        # Check if the casts cancel out (same input and output type)
        if from_type == to_type:
            # Remove both casts, connect input directly to final output
            original_input = first_node.input[0]
            final_output = second_node.output[0]
            output_replacements[final_output] = original_input
            nodes_to_remove.add(first_idx)
            nodes_to_remove.add(second_idx)

            if verbose:
                print("      -> Removing redundant cast chain (same input/output type)")
        else:
            # Combine into single cast
            original_input = first_node.input[0]
            final_output = second_node.output[0]

            # Update the second cast to take the original input
            second_node.input[0] = original_input

            # Remove the first cast
            nodes_to_remove.add(first_idx)

            if verbose:
                print("      -> Combined into single cast")

    # Create new node list without removed nodes
    new_nodes = []
    for i, node in enumerate(graph.node):
        if i not in nodes_to_remove:
            # Update any inputs that reference removed outputs
            new_node = onnx.NodeProto()
            new_node.CopyFrom(node)

            for j, input_name in enumerate(new_node.input):
                if input_name in output_replacements:
                    new_node.input[j] = output_replacements[input_name]

            new_nodes.append(new_node)

    # Update graph outputs if they reference removed nodes
    for i, output in enumerate(graph.output):
        if output.name in output_replacements:
            graph.output[i].name = output_replacements[output.name]

    # Clear and rebuild the node list
    del graph.node[:]
    graph.node.extend(new_nodes)

    if verbose:
        print(f"  Removed {len(nodes_to_remove)} cast nodes from this graph")

    return True  # Graph was modified


def find_back_to_back_casts_in_model(
    model: onnx.ModelProto,
) -> List[Tuple[str, List[Tuple[int, int, str, int, int]]]]:
    """
    Find back-to-back Cast operations in the entire model, including subgraphs.

    Returns:
        List of tuples: (graph_location, back_to_back_casts_list)
    """
    all_casts = []

    # Process main graph
    main_casts = find_back_to_back_casts(model.graph)
    if main_casts:
        all_casts.append(("main graph", main_casts))

    # Process subgraphs
    def find_in_subgraphs(graph: onnx.GraphProto, location_prefix: str):
        for node_idx, node in enumerate(graph.node):
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.GRAPH:
                    subgraph_location = (
                        f"{location_prefix}.{node.op_type}[{node_idx}].{attr.name}"
                    )
                    subgraph_casts = find_back_to_back_casts(attr.g)
                    if subgraph_casts:
                        all_casts.append((subgraph_location, subgraph_casts))
                    # Recursively check nested subgraphs
                    find_in_subgraphs(attr.g, subgraph_location)
                elif attr.type == onnx.AttributeProto.GRAPHS:
                    for i, subgraph in enumerate(attr.graphs):
                        subgraph_location = f"{location_prefix}.{node.op_type}[{node_idx}].{attr.name}[{i}]"
                        subgraph_casts = find_back_to_back_casts(subgraph)
                        if subgraph_casts:
                            all_casts.append((subgraph_location, subgraph_casts))
                        # Recursively check nested subgraphs
                        find_in_subgraphs(subgraph, subgraph_location)

    find_in_subgraphs(model.graph, "main graph")

    return all_casts


def main():
    parser = argparse.ArgumentParser(
        description="Remove back-to-back Cast operations from ONNX models"
    )
    parser.add_argument("input_model", help="Path to input ONNX model")
    parser.add_argument("output_model", help="Path to output ONNX model")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Print verbose information"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only check for back-to-back casts, don't modify",
    )

    args = parser.parse_args()

    try:
        # Load the model
        print(f"Loading model from {args.input_model}")
        model = onnx.load(args.input_model)

        # Verify the model
        onnx.checker.check_model(model)

        if args.check:
            # Just check for back-to-back casts in the entire model
            all_casts = find_back_to_back_casts_in_model(model)
            if all_casts:
                total_casts = sum(len(casts) for _, casts in all_casts)
                print(
                    f"Found {total_casts} back-to-back cast pairs across {len(all_casts)} graph(s):"
                )
                for location, casts in all_casts:
                    print(f"\nIn {location}:")
                    for _, _, _, from_type, to_type in casts:
                        print(
                            f"  Cast {get_data_type_name(from_type)} -> {get_data_type_name(to_type)}"
                        )
            else:
                print("No back-to-back casts found in the entire model.")
            return

        # Remove back-to-back casts
        optimized_model = remove_back_to_back_casts(model, verbose=args.verbose)

        # Verify the optimized model
        onnx.checker.check_model(optimized_model)

        # Save the optimized model
        print(f"Saving optimized model to {args.output_model}")
        onnx.save(optimized_model, args.output_model)

        print("Done!")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
