import argparse
import configparser


class DefenseConfig(object):
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.config_parser = configparser.ConfigParser()
        self.config_section = self.args.config_section
        self.converters = None

    def load_config(self):
        converters = self.converters
        if converters is None:
            converters = {}

        # Read the configuration file
        self.config_parser.read(self.args.config_path)

        # Check if the specified section exists in the configuration file
        if not self.config_parser.has_section(self.config_section):
            raise ValueError(f"Section '{self.config_section}' not found in the configuration file.")

        # Get all options and their values in the specified section
        options = self.config_parser.options(self.config_section)

        required = set(converters.keys())
        present = set(options)
        missing = sorted(required - present)
        unexpected = sorted(present - required)
        if missing:
            defense_hint = self.__class__.__name__.replace("Config", "").lower()
            raise ValueError(
                f"Config mismatch in {self.args.config_path!r} section [{self.config_section!r}]: "
                f"missing required option(s) {missing}. "
                f"You are probably using the wrong .ini for this defense "
                f"(e.g. use defenses/config/{defense_hint}.ini for {defense_hint})."
            )
        if unexpected:
            raise ValueError(
                f"Config mismatch in {self.args.config_path!r} section [{self.config_section!r}]: "
                f"unexpected option(s) {unexpected} (this defense expects exactly {sorted(required)})."
            )

        # Populate self.raw_config with the options and their values
        for option in options:
            raw_value = self.config_parser.get(self.config_section, option)
            # Use the specified converter or the default str() conversion
            converter = converters.get(option, str)
            setattr(self, option, converter(raw_value))


class ChameleonConfig(DefenseConfig):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.converters = {
            'trace_threshold': int,
            'selection_k': int,
            'selection_ratio': float,
            'selection_min': int,
            'selection_alpha': float,
            'selection_beta': float,
            'selection_gamma': float,
            'selection_seq_len': int,
            'radix_trie_build_length': int,
            'mutation': int,
            'mutation_length': int,
        }
