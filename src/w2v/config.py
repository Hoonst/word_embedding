import argparse


def load_config():
    parser = argparse.ArgumentParser()

    # default hparams
    parser.add_argument("--dset", type=str, default="cifar100")
    parser.add_argument("--dpath", type=str, default="src/word2vec/data/text8")
    parser.add_argument("--ckpt-path", type=str, default="checkpoints/")
    parser.add_argument("--emb_dimension", type=int, default=300)
    parser.add_argument("--window_size", type=int, default=5)

    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")
    parser.add_argument("--eval-step", type=int, default=100)
    parser.add_argument("--contain-test", action="store_true", default=False)

    # training hparams
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--model", type=str, default="w2v")

    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--lr-decay-step-size", type=int, default=60)
    parser.add_argument("--lr-decay-gamma", type=float, default=0.1)

    args = parser.parse_args()
    return args
