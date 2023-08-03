import signal
import sys
from argparse import ArgumentParser, Namespace
from types import FrameType
from typing import Optional


from config import Config
from input import Input
from output import Output
from vnf_placement import MultiFlavouredVNFChainPlacement
from genetic_placement_searcher import (
    search_optimal_placement,
)


def main() -> None:
    _register_signal_handlers()
    args = _parse_command_line_arguments()
    config = _load_config(args.config_file_path)
    input_ = _read_input(args.input_file_path)
    placement = search_optimal_placement(input_, config)
    _write_output(args.output_file_path, placement)


def _register_signal_handlers() -> None:
    signalnums = [signal.SIGINT, signal.SIGTERM]

    def signal_handler(_signalnum: int, _frame: Optional[FrameType]) -> None:
        sys.exit()

    for signalnum in signalnums:
        signal.signal(signalnum, signal_handler)


def _parse_command_line_arguments() -> Namespace:
    description = (
        "search for efficient placements of Virtual Network Functions within a network "
        + "using a Genetic Algorithm"
    )
    arg_parser = ArgumentParser(description=description, add_help=False)
    arg_parser.add_argument(
        "input_file_path",
        metavar="INPUT_FILE",
        type=str,
        help="path to the input file to be processed",
    )
    required_arguments = arg_parser.add_argument_group("required arguments")
    required_arguments.add_argument(
        "-c",
        "--config",
        dest="config_file_path",
        type=str,
        required=True,
        help="path to the configuration file",
    )
    required_arguments.add_argument(
        "-o",
        "--output",
        dest="output_file_path",
        type=str,
        required=True,
        help="path to the output file where results will be saved",
    )
    optional_arguments = arg_parser.add_argument_group("optional arguments")
    optional_arguments.add_argument(
        "-h", "--help", action="help", help="show this help message and exit"
    )
    args = arg_parser.parse_args()
    return args


def _load_config(config_file_path: str) -> Config:
    return Config.from_file(config_file_path)


def _read_input(input_file_path: str) -> Input:
    return Input.from_file(input_file_path)


def _write_output(
    output_file_path: str, placement: Optional[MultiFlavouredVNFChainPlacement]
) -> None:
    output = Output(placement)
    output.to_file(output_file_path)


if __name__ == "__main__":
    main()
