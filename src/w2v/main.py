import glob
import os
import time
import torch
import torch.nn as nn

from config import load_config

from net import SkipGramNeg, NegativeSamplingLoss
import torch.optim as optim
import dataset

from tensorboardX import SummaryWriter
from tqdm import tqdm


def logging_time(original_fn):
    def wrapper_fn(*args, **kwargs):
        start_time = time.time()
        result = original_fn(*args, **kwargs)
        end_time = time.time()
        print(
            "WorkingTime[{}]: {} sec".format(
                original_fn.__name__, end_time - start_time
            )
        )
        return result

    return wrapper_fn


@logging_time
def main(hparams):
    with open("data/text8") as f:
        text = f.read()

    words = dataset.preprocess(text)

    vocab_to_int, int_to_vocab = dataset.create_lookup_tables(words)
    int_words = [vocab_to_int[word] for word in words]

    train_words, noise_dist = dataset.subsampling(int_words)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    embedding_dim = 300
    model = SkipGramNeg(len(vocab_to_int), embedding_dim, noise_dist=noise_dist).to(
        device
    )

    # using the loss that we defined
    criterion = NegativeSamplingLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    print_every = 1500
    epochs = hparams.epochs
    batch_size = hparams.batch_size

    version = 0

    while True:
        save_path = os.path.join(hparams.ckpt_path, f"version-{version}")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            break
        else:
            version += 1

    summarywriter = SummaryWriter(save_path)
    global_step = 0
    for epoch in range(epochs):

        for step, (input_words, target_words) in tqdm(
            enumerate(dataset.get_batches(train_words, batch_size)),
            desc="Training On!",
            total=len(train_words) // batch_size,
            # total= len(train_words[:(len(train_words)//batch_size)])
        ):
            global_step += 1
            # steps+=1
            inputs, targets = torch.LongTensor(input_words), torch.LongTensor(
                target_words
            )
            inputs, targets = inputs.to(device), targets.to(device)

            # input, outpt, and noise vectors
            input_vectors = model.forward_input(inputs)
            output_vectors = model.forward_output(targets)
            noise_vectors = model.forward_noise(inputs.shape[0], 5)

            # negative sampling loss
            loss = criterion(input_vectors, output_vectors, noise_vectors)
            summarywriter.add_scalars("loss", {"train": loss}, global_step)
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            if step % print_every == 0:
                tqdm.write(f"Epoch: {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
                # save model
                new_path = os.path.join(
                    save_path, f"best_model_epoch_{epoch}_acc_{loss.item():.4f}.pt"
                )

                for filename in glob.glob(os.path.join(save_path, "*.pt")):
                    os.remove(filename)
                torch.save(model.state_dict(), new_path)
                summarywriter.close()

            # print("Epoch: {}/{}".format(e+1, epochs))
            # print("Loss: ", loss.item()) # avg batch loss at this point in training
            # valid_examples, valid_similarities = dataset.cosine_similarity(model.in_embed, device=device)
            # _, closest_idxs = valid_similarities.topk(6)

            # valid_examples, closest_idxs = valid_examples.to('cpu'), closest_idxs.to('cpu')
            # for ii, valid_idx in enumerate(valid_examples):
            #     closest_words = [int_to_vocab[idx.item()] for idx in closest_idxs[ii]][1:]
            #     print(int_to_vocab[valid_idx.item()] + " | " + ', '.join(closest_words))
            # print("...\n")


if __name__ == "__main__":
    hparams = load_config()
    main(hparams)
