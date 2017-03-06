#!/usr/bin/env python

"""Evaluate a neural network using OpenStreetMap labels and NAIP images."""

import argparse
from src.training_visualization import render_result_jpegs


def create_parser():
    """Create the argparse parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--neural-net",
                        default='one_layer_relu',
                        choices=['one_layer_relu', 'one_layer_relu_conv', 'two_layer_relu_conv'],
                        help="the neural network architecture to use")
    parser.add_argument("--no-render-results",
                        action='store_false',
                        dest='render_results',
                        default=True,
                        help="output data/predictions to JPEG, in addition to normal JSON")
    return parser


def main():
    """Use local data to train the neural net, probably made by bin/create_training_data.py."""
    parser = create_parser()
    args = parser.parse_args()
    # TODO: numerical evaluation should go here
    if args.render_results:
        render_result_jpegs(args.neural_net)


if __name__ == "__main__":
    main()
