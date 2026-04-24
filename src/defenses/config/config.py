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


class FrontConfig(DefenseConfig):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.converters = {
            'n_client': int,
            'n_server': int,
            'w_min': float,
            'w_max': float,
            'start_t': float
        }


class TamarawConfig(DefenseConfig):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.converters = {
            'rho_client': float,
            'rho_server': float,
            'lpad': int,
            'strategy': str
        }


class RegulatorConfig(DefenseConfig):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.converters = {
            'initial_rate': int,
            'decay_rate': float,
            'surge_threshold': float,
            'max_padding_budget': int,
            'upload_ratio': float,
            'delay_cap': float
        }


class WtfpadConfig(DefenseConfig):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.converters = {}
        self.interpolate = True
        self.remove_tokens = True
        self.stop_on_real = True
        self.percentile = 0


class TrafficSliverConfig(DefenseConfig):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.converters = {
            'n_circuits': int,
            'batch_size_min': int,
            'batch_size_max': int,
            'latency_file_path': str,
            'strategy': str
        }


def _config_bool(s):
    v = str(s).strip().lower()
    if v in ('true', 'yes', '1', 'ture'):  # tolerate common ini typo
        return True
    if v in ('false', 'no', '0'):
        return False
    raise ValueError(f"Invalid boolean config value: {s!r}")


class MinipatchConfig(DefenseConfig):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.converters = {
            'patches': int,
            'inbound': int,
            'outbound': int,
            'maxiter': int,
            'maxquery': int,
            'adaptive': _config_bool,
            'threshold': float,
            'polish': _config_bool,
            'initial_temp': float,
            'restart_temp_ratio': float,
            'visit': float,
            'accept': float,
        }


class DynaflowConfig(DefenseConfig):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.converters = {
            'first_time_gap': float,
            'subseq_length': int,
            'memory': int,
            'switch_sizes': str,  # Comma-separated string of integers
            'poss_time_gaps': str,  # Comma-separated string of floats
            'm': float
        }


class PaletteConfig(DefenseConfig):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.converters = {
            'tam_length': int,
            'cutoff_time': float,
            'round': int,
            'set_size': int,
            'alpha_upload': float,
            'alpha_download': float,
            'u_upload': int,
            'u_download': int,
            'b': int,
            'seed': int,
            'lr': float,
            'batch_size': int,
            'k': int,
            'num_epochs': int,
        }


class MockingbirdConfig(DefenseConfig):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.converters = {
            'max_bursts': int,
            'num_iterations': int,
            'alpha': float,
            'target_pool_size': int
        }


def parse_comma_separated_ints(s):
    """Parse comma-separated string into list of ints. Picklable for multiprocessing."""
    return [int(x.strip()) for x in s.split(',') if x.strip()]

def parse_int_or_first_csv(s):
    """
    Parse an integer config value, tolerating accidental comma-separated input.
    If multiple values are provided, the first one is used.
    """
    vals = parse_comma_separated_ints(s)
    if not vals:
        raise ValueError(f"Invalid integer config value: {s!r}")
    return int(vals[0])


class GapdisConfig(DefenseConfig):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.converters = {
            'max_perturb': int,
            'max_iterations': int,
            'target_acc': float,
            'topk_num': int,
            'tabu_len_multi': int,
            'max_dummy': int,
            'sol_len_multi': int,
            'cpm_len_den': int,
            'init_rd_num': int,
            'init_m_num': int,
            'exch_len': int,
            'repl_rate': float,
            'muta_rate': float,
            'smp_cpm_rate': float,
            'toler': int,
            'max_iter_multi': int,
        }


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


class SurakavConfig(DefenseConfig):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.converters = {
            'model_dir': str,
            'dummy_code': int,
            'seed': int,
        }


class AlertConfig(DefenseConfig):
    """ALERT GAN-style defense: burst perturbation with per-class generators."""

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.converters = {
            'model_dir': str,
            'max_length': int,
            'oh_min_threshold': float,
            'oh_max_threshold': float,
        }


if __name__ == '__main__':
    # Example usage:
    args = argparse.Namespace(config_section='default',
                              config_path='./trafficsliver.ini')
    defense_config = TrafficSliverConfig(args)
    defense_config.load_config()
    print(defense_config)
