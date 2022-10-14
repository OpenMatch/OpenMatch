import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str)
    parser.add_argument("output_file", type=str)
    parser.add_argument("--prefix", type=str, default="dev")
    args = parser.parse_args()

    with open(args.input_file, "r") as fin, open(args.output_file, "w") as fout:
        for i, line in enumerate(fin):
            fout.write(f"{args.prefix}_{i}\t{line}")