import torch, os
import torch.nn as nn
from classes import *
from torch.utils.data import DataLoader


def predict_author(focus_context_embedding, focus_check_embedding):
    print("starting function: predict_author() in decision.py")
    input_size = 58
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    new_nhead = 8  # 16  # 8  # 2 has been optimal so far btw
    new_dim_feedforward = 32  # should be 32
    new_dropout = 0.39472428838464607

    siamese_net = SiameseNetwork(
        input_size, new_nhead, new_dim_feedforward, new_dropout).to(device)
    output_dim = siamese_net.embedding_net.get_output_dim()
    classifier_net = ClassifierNet(output_dim).to(
        device)  # Make sure output_dim is defined

    new_checkpoint_path = "model.pth"
    print('loading model')

    new_checkpoint = torch.load(new_checkpoint_path, map_location=device)
    print('model loaded')
    # Load the state dictionaries
    siamese_net.load_state_dict(new_checkpoint['siamese_model_state_dict'])
    # Get the output dimension of EmbeddingNet
    classifier_net.load_state_dict(new_checkpoint['classifier_model_state_dict'])

    # Usage example
    # filler_csv_paths = ["5_3.csv", "FGPT4-2.csv", "VTL10_Gen2.csv", "VTL10_Gen3.csv",
    #                   # Add your CSV file paths here
    #                  "VTL20_Gen3_t.csv", "4-2_2.csv", "4-2_1.csv", "321KGPT3-2-3.csv"]

    # filler_csv_paths = ["5_3.csv", "FGPT4-2.csv", "VTL10_Gen2.csv", "VTL10_Gen3.csv",
    #                    "VTL20_Gen3_t.csv", "4-2_2.csv", "4-2_1.csv", "321KGPT3-2-3.csv", "VTL20_Gen4_t.csv", "VTL20_Gen5_t.csv"]

    # Specify the directory where your CSV files are located
    #csv_files_dir = r'C:\Users\S\Desktop\VerifyMe\Op4_Workspace 12.2.2024\archive_name'

   # filler_csv_paths = [
   #     "5_3.csv",
   #     "FGPT4-2.csv",
   #     "4-2_2.csv",
   #     "4-2_1.csv",
   #     "321KGPT3-2-3.csv",
   #     "VTL10_Gen2_test.csv",
   #     "VTL10_Gen3_test.csv",
   #     "VTL20_Gen3_t.csv",
   #     "VTL20_Gen4_t.csv",
   #     "VTL20_Gen5_t.csv"
   # ]



 #   filler_csv_paths = [
 #       "VTL20_Gen3_test.csv",
 #   ]


    csv_files_dir = r'/Users/sean/Desktop/vscode/app/fillerCsvs'

    filler_csv_paths = [
        "5_3.csv",
        "FGPT4-2.csv",
        "4-2_2.csv",
        "4-2_1.csv",
        "3-1.5-3.csv",
        "GPT4-1_3.csv",
        "321KGPT3-2-3.csv",
        "VTL10_Gen2_test.csv",
        "VTL10_Gen3_test.csv",
        "VTL20_Gen2_t.csv",
        "VTL20_Gen3_t.csv",
        "VTL20_Gen4_t.csv",
        "5_3.csv",
        "FGPT4-2.csv",
        "4-2_2.csv",
        "4-2_1.csv",
        "3-1.5-3.csv",
        "GPT4-1_3.csv",
        "321KGPT3-2-3.csv",
        "VTL10_Gen2_test.csv",
        "VTL10_Gen3_test.csv",
        "VTL20_Gen2_t.csv",
        "VTL20_Gen3_t.csv",
        "VTL20_Gen4_t.csv",
        "VTL20_Gen5_t.csv"

    ]

    # Prepend the directory path to each CSV file name
    filler_csv_paths = [os.path.join(csv_files_dir, file_name) for file_name in filler_csv_paths]

    e = len(filler_csv_paths)
    print(e)
    percentages = [1/e for i in range(0,e)]
    print(percentages)

    #percentages = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    #percentages = [1]

    batch_size = 1024  # Set your desired batch size

    print("Creating the custom dataset now")
    custom_dataset = CustomDataset(filler_csv_paths, percentages, focus_context_embedding, focus_check_embedding, batch_size)
    print("Now creating the test dataloader")

    test_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=False)
    print("Created the test loader.")

    print("now I am about to try iterate through the test loader we just made")
    # Set the print options to display the full array
    np.set_printoptions(threshold=np.inf)

    #for batch_idx, (context, check, label, is_focus) in enumerate(test_loader):
        # Convert is_focus to a NumPy array if it's not already one and print it
        #is_focus_arr = is_focus.numpy() if hasattr(is_focus, 'numpy') else is_focus
        #print(f"Batch {batch_idx}: Is Focus:", is_focus_arr)
        #label_arr = label.numpy() if hasattr(label, 'numpy') else label
        #print(f"Batch {batch_idx}: Label:", label_arr)
        #context_arr = context.numpy() if hasattr(context, 'numpy') else context
        #print(f"Batch {batch_idx}: Label:", context_arr)
        #check_arr = check.numpy() if hasattr(check, 'numpy') else check
        #print(f"Batch {batch_idx}: Label:", check_arr)

    criterion = nn.MSELoss()

    #print('just done printing check array, the last item in that array should have been the check embedding generated in our flask web app')
    #print('from my observations, during the printout the context embedding was also all zeroes')
    print("just about to predict authorship in decision.py")
    
    #there is some issue with the testloader in its current form. 
    #focus_prediction, accuracy_all = evaluate2(siamese_net, classifier_net, test_loader, criterion, device)
    focus_prediction = evaluate2(siamese_net, classifier_net, test_loader, criterion, device)

    #return f"accuracy_all_filler={accuracy_all}, focus_prediction={focus_prediction}"
    #return f"{accuracy_all}:{focus_prediction}"
    return f"{focus_prediction}"