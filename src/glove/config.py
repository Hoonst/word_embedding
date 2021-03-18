import argparse


def load_config():
    parser = argparse.ArgumentParser()

    # default hparams
    parser.add_argument("--dpath", type=str, default="data/text8")
    parser.add_argument("--ckpt-path", type=str, default="checkpoints/")
    parser.add_argument("--emb_dimension", type=int, default=300)
    parser.add_argument("--window_size", type=int, default=5)
    parser.add_argument("--n_words", type=int, default=100000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument("--X_MAX", type=int, default=100)
    parser.add_argument("--ALPHA", type=int, default=0.75)

    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")
    parser.add_argument("--eval-step", type=int, default=100)
    parser.add_argument(
        "--amp", action="store_true", default=False, help="PyTorch(>=1.6.x) AMP"
    )
    parser.add_argument("--contain-test", action="store_true", default=False)

    # training hparams
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--model", type=str, default="glove")

    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--lr-decay-step-size", type=int, default=60)
    parser.add_argument("--lr-decay-gamma", type=float, default=0.1)

    args = parser.parse_args()
    return args
