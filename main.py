import argparse
from source.train.train import test_multitask, test_multitask_ensemble, train_multitask
from source.util.utils import seed_everything


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sst_train", type=str, default="data/ids-sst-train.csv")
    parser.add_argument("--sst_dev", type=str, default="data/ids-sst-dev.csv")
    parser.add_argument("--sst_test", type=str, default="data/ids-sst-test-student.csv")

    parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")

    parser.add_argument("--sts_train", type=str, default="data/sts-train.csv")
    parser.add_argument("--sts_dev", type=str, default="data/sts-dev.csv")
    parser.add_argument("--sts_test", type=str, default="data/sts-test-student.csv")

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--recover", type=bool, default=False)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--epochs_sts", type=int, default=20)
    parser.add_argument("--epochs_paraphrase", type=int, default=10)
    parser.add_argument("--epochs_sst", type=int, default=20)
    parser.add_argument("--fine-tune-mode", type=str,
                        help='last-linear-layer: the BERT parameters are frozen and the task specific head parameters are updated; full-model: BERT parameters are updated as well',
                        choices=('last-linear-layer', 'full-model'), default="last-linear-layer")
    parser.add_argument("--use_gpu", action='store_true')

    parser.add_argument("--sst_dev_out", type=str, default="predictions/sst-dev-output.csv")
    parser.add_argument("--sst_test_out", type=str, default="predictions/sst-test-output.csv")

    parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

    parser.add_argument("--sts_dev_out", type=str, default="predictions/sts-dev-output.csv")
    parser.add_argument("--sts_test_out", type=str, default="predictions/sts-test-output.csv")

    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=64)
    parser.add_argument("--batch_size_para", help='paraphrase: 64', type=int, default=64)
    parser.add_argument("--batch_size_sts", help='similarity: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=64)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--hidden_dropout_prob_sst", type=float, default=0.3)
    parser.add_argument("--hidden_dropout_prob_para", type=float, default=0.3)
    parser.add_argument("--hidden_dropout_prob_sts", type=float, default=0.1)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
    parser.add_argument("--ensemble",help="enable the ensemble function", type=bool, default=False)
    parser.add_argument("--ensemble_model_count", help="ensemble model count", type=int, default=3)
    parser.add_argument("--num_batch_per_epoch", type=int, default=256)

    parser.add_argument("--use_cuda", action='store_true')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    seed_everything(args.seed)  # Fix the seed for reproducibility.

    if args.ensemble:
        ensemble_model_counts = args.ensemble_model_count
        filePaths = []
        for count in range(ensemble_model_counts):
            args.filepath = f'{args.fine_tune_mode}-{args.epochs}-{args.lr}-ensemble-{count}-multitask.pt' # Save path.
            filePaths.append(args.filepath)
            train_multitask(args)
        test_multitask_ensemble(args, filePaths)
    else:
        args.filepath = f'{args.fine_tune_mode}-{args.epochs}-{args.lr}-multitask.pt' # Save path.
        train_multitask(args)
        test_multitask(args)