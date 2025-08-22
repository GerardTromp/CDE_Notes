import argparse


class KeyValueAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        items = getattr(namespace, self.dest, {}) or {}
        for item in values:  # type: ignore
            if "=" not in item:
                raise argparse.ArgumentTypeError(
                    f"Argument '{item}' must be in KEY=VALUE format."
                )
            key, value = item.split("=", 1)
            items[key] = value
        setattr(namespace, self.dest, items)


def cde_argparse():
    """
    Perform the argument parsing
    Separating this out permits easy extension of options
    """
    parser = argparse.ArgumentParser("Arguments for CDE domain clustering and plotting")

    parser.add_argument(
        "--umapkwargs",
        nargs="*",
        action=KeyValueAction,
        help="Add UMAP keyword-value pairs as param=setting(e.g., epochs=300 metric=euclidian ...)",
    )
    parser.add_argument(
        "--tsnekwargs",
        nargs="*",
        action=KeyValueAction,
        help="Add TSNE keyword-value pairs as param=setting(e.g., perplexit=30 metric=euclidian ...)",
    )
    parser.add_argument(
        "--config-path",
        nargs=1,
        type=str,
        action="store",
        default="config/cde_analysis.ini",
        help="Path to an 'ini' format file containing the analysis data",
    )
    parser.add_argument(
        "--param-path",
        nargs=1,
        type=str,
        action="store",
        default="config/parameters.ini",
        help="Path to an 'ini' format configuration file with the parameters for analysis algorithms",
    )

    args = parser.parse_args()

    return args
