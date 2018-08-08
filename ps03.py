import task1, task2, task3, argparse
import numpy as np

parser = argparse.ArgumentParser(description = "task 2 predictions, run as python task2_submission.py -t 'train|test' -d 'data.txt'")
parser.add_argument("-t", help = "either 'test' to get predictions, or 'train' to train and save a model", required = True)
parser.add_argument("-d", help = "data file to use the model on", required = True)
args = parser.parse_args()


def main():
    data = [line.strip().split("\t") for line in open(args.d)]
    if args.t == "train":
        task1.train_model(data)
        task2.train_model_t2(data)
        #data = [line.strip().lower().split("\t") for line in open(args.d)]
        task3.train(data)
    elif args.t == "test":
        y_t1 = np.array(task1.get_predictions(data))
        y_t2 = np.array(task2.get_predictions_t2(data))
        y_t3 = np.array(task3.getPrediction(data))

        id_sent = np.array([el[0:1] for el in data])
        
        pred = np.column_stack((id_sent, y_t1, y_t2, y_t3))
        pred = ["\t".join(el) for el in pred]

        out_fname = "ps3_predictions.txt"
        out_file = open(out_fname, "w")
        for i in range(0, len(data)):
            out_file.writelines(data[i][0] + "\t" + data[i][1] + "\t" + y_t2[i] + "\t" + y_t3[i] + "\t" + y_t1[i] + "\n")

    else:
        print("invalid option for -t, use 'train' or 'test'")


if __name__ == "__main__":
    main()
