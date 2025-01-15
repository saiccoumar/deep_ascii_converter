# %%
import subprocess
import csv
import re

factors = [2.5, 2.5, 1.5, 2.5, 1.5,
            2, 2, 2, 2, 1.5,
            2, 2, 2, 2, 3, 
            2, 2, 3.5, 2, 3]

for image_num in range(1,len(factors)+1):
    factor = str(factors[image_num-1])

    # %%
    # Define the input file and algorithms
    if image_num < 10:
        input_file = f"./oscii_data/images/0{image_num} original.png"
    else: 
        input_file = f"./oscii_data/images/{image_num} original.png"
    algorithms = ["aiss", "nn", "rforest", "svm", "knn", "mobile", "resnet", "cnn"]
    extended_algorithms = algorithms + ["knn_no_hog", "svm_no_hog", "rforest_no_hog", "knn_ae"]
    outfile = "conversion_times.csv"

    # %%
    # Regex to extract model conversion time from script output
    time_pattern = re.compile(r"Model Conversion Time: ([\d.]+) seconds")

    # Dictionary to store results
    results = {algo: None for algo in algorithms}


    # %%
    # Run the script for each algorithm and extract the model conversion time
    for algo in algorithms:
        print(f"converting img {image_num} with {algo}...")
        try:
            result = subprocess.run(
                ["python", "./convert_static.py", "--filename", input_file, '-ded', "--algorithm", algo, "-f", factor],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            if result.returncode != 0:
                print(f"Error with algorithm {algo}: {result.stderr}")
                continue

            # Extract the model conversion time
            match = time_pattern.search(result.stdout)
            if match:
                results[algo] = float(match.group(1))
            else:
                print(f"Model conversion time not found for {algo}")
        except Exception as e:
            print(f"Exception occurred for algorithm {algo}: {e}")

    # %%
    # Run the script for additional algorithms with --disable_hog flag
    for algo in ["knn", "svm", "rforest"]:
        print(f"converting img {image_num} with {algo} no hog ...")
        try:
            algo_no_hog = f"{algo}_no_hog"
            result = subprocess.run(
                ["python", "./convert_static.py", "--filename", input_file, '-ded', "--algorithm", algo, "--disable_hog", "-f", factor],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            if result.returncode != 0:
                print(f"Error with algorithm {algo_no_hog}: {result.stderr}")
                continue

            # Extract the model conversion time
            match = time_pattern.search(result.stdout)
            if match:
                results[algo_no_hog] = float(match.group(1))
            else:
                print(f"Model conversion time not found for {algo_no_hog}")
        except Exception as e:
            print(f"Exception occurred for algorithm {algo_no_hog}: {e}")


    # %%
    # Run the script for knn with --disable_hog and -ae flag
    
    try:
        algo_no_hog_ae = "knn_ae"
        print(f"Converting img {image_num} with {algo_no_hog_ae}...")
        
        result = subprocess.run(
            ["python", "./convert_static.py", "--filename", input_file, "--algorithm", "knn", '-ded', "--disable_hog", "-ae", "-f", factor],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if result.returncode != 0:
            print(f"Error with algorithm {algo_no_hog_ae}: {result.stderr}")
        else:
            # Extract AE Conversion Time and Model Conversion Time
            ae_time_pattern = re.compile(r"AE Conversion Time: ([\d.]+) seconds")
            model_time_pattern = re.compile(r"Model Conversion Time: ([\d.]+) seconds")
            
            ae_match = ae_time_pattern.search(result.stdout)
            model_match = model_time_pattern.search(result.stdout)
            
            if ae_match and model_match:
                ae_time = float(ae_match.group(1))
                model_time = float(model_match.group(1))
                total_time = ae_time + model_time
                results[algo_no_hog_ae] = total_time
                # print(f"Total time for {algo_no_hog_ae}: {total_time:.4f} seconds")
            else:
                print(f"Conversion times not found for {algo_no_hog_ae}")
    except Exception as e:
        print(f"Exception occurred for algorithm {algo_no_hog_ae}: {e}")

    # %%

    # Write results to CSV
    with open(outfile, mode="a", newline="") as csvfile:
        writer = csv.writer(csvfile)

        # If the file is empty, write the header first
        try:
            if csvfile.tell() == 0:  # Check if the file is empty
                header = ["Filename"] + extended_algorithms
                writer.writerow(header)
        except Exception as e:
            print(f"Error checking CSV file state: {e}")

        # Write times
        row = [input_file] + [results[algo] if results[algo] is not None else "N/A" for algo in extended_algorithms]
        writer.writerow(row)

    print(f"Results appended to {outfile}")

