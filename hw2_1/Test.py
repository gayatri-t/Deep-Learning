import sys
import torch
import json
from torch.utils.data import DataLoader
import pickle
import model as custom_model
import Train as custom_train
import bleu_eval as custom_bleu_eval
from loss_calculator import LossCalculator, SimpleModel
from torch.optim import SGD
import torch.nn as nn


def main():
    test_input = sys.argv[1]
    test_json_path = "./testing_label.json"
    seq2seqmodel_path = "./seq2seqModel.h5"
    output_file_path = sys.argv[2]
    
    trained_model = torch.load(seq2seqmodel_path)
    
    data_directory = test_input
    i2w, w2i, dictionary = custom_train.create_dictionary(4)
    print("Dictionary length: " + str(len(dictionary)))

    test_dataset = custom_train.test_data_loader(data_directory)

    test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True, num_workers=8)
    
    model = trained_model
    predictions = custom_train.test_function(test_dataloader, model, i2w)
    
    try:
        with open(output_file_path, 'w') as output_file:
            for identifier, prediction in predictions:
                output_file.write('{},{}\n'.format(identifier, prediction))
            print('All Files opened successfully!')
    except FileNotFoundError:
        with open(output_file_path, 'x') as output_file:
            for identifier, prediction in predictions:
                output_file.write('{},{}\n'.format(identifier, prediction))
            print('File created and updated successfully!')

    # BLEU Eval
    test_data = json.load(open(test_json_path, 'r'))
    output_path = output_file_path
    result_dict = {}

    with open(output_path, 'r') as output_file:
        for line in output_file:
            line = line.rstrip()
            comma_index = line.index(',')
            test_id = line[:comma_index]
            caption = line[comma_index + 1:]
            result_dict[test_id] = caption

    bleu_scores = []
    ground_truths = {}
    for item in test_data:
        scores_per_video = []
        captions = [x.rstrip('.') for x in item['caption']]
        scores_per_video.append(custom_bleu_eval.BLEU(result_dict[item['id']], captions, True))
        bleu_scores.append(scores_per_video[0])

    average_bleu = sum(bleu_scores) / len(bleu_scores)
    print("Average BLEU score is {:.6f}".format(average_bleu))
    
    input_size = 10
    output_size = 5
    input_data = torch.randn((100, input_size))
    target_data = torch.randint(0, output_size, (100,)).long()

    # Create model, optimizer, and loss function
    model = SimpleModel(input_size, output_size)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Create LossCalculator instance
    loss_calculator = LossCalculator(model, optimizer, loss_fn)

    # Train the model and get the average loss
    average_loss = loss_calculator.train_model(input_data, target_data)

    # Print the average loss
    print(f"Average Loss is {average_loss:.6f}")
        
if __name__ == '__main__':
    main()
