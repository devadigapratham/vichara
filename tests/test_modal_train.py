from scripts.modal_train import TINYSTORIES_FULL, command
from training.pretrain import parse_args


def test_command_always_resumes():
    assert command("--max-steps 10")[-1] == "--resume"


def test_default_flags_parse_into_a_valid_config():
    cfg, resume = parse_args(command(TINYSTORIES_FULL))
    assert resume is True
    assert cfg.dataset == "tinystories"
    assert cfg.out_dir == "checkpoints/tinystories_full"
    assert cfg.precision == "bf16" and cfg.sdpa and cfg.compile
    assert cfg.max_steps * cfg.batch_size * cfg.seq_len > 280_000_000      # ~Chinchilla for 14M params


def test_quoted_flags_split_like_a_shell():
    assert command("--out-dir 'checkpoints/my run'")[:2] == ["--out-dir", "checkpoints/my run"]
