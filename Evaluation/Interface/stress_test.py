import requests
import concurrent.futures
import csv
import time
import os

# URL for the submit endpoint
url = "http://127.0.0.1:5000/submit"

# Sample data to be sent in the requests
data_template = {
    "prompt": "Generate a landscape image",
    "num_images": 1,
    "scheduler": "DDIM",
    "inference_steps": 50
}

def send_request(prompt_num):
    """
    Sends a POST request to the Flask server with a modified prompt and measures the response time.
    
    Args:
        prompt_num (int): The number to append to the prompt for uniqueness.
        
    Returns:
        dict: A dictionary containing the JSON response from the server and the response time.
    """
    data = data_template.copy()
    data["prompt"] = f"Generate a landscape image {prompt_num}"
    
    start_time = time.time()
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()  # Raise an error for bad status codes
        end_time = time.time()
        response_time = end_time - start_time
        return {"response": response.json(), "response_time": response_time}
    except requests.exceptions.RequestException as e:
        end_time = time.time()
        response_time = end_time - start_time
        return {"response": {"error": str(e)}, "response_time": response_time}

def main():
    """
    Main function to conduct the stress test by sending multiple prompts concurrently.
    
    The test sends 5, 10, 15, and 20 simultaneous prompts to the server, measures response times,
    and saves the responses and metrics to a CSV file.
    """
    # Number of prompts to send in each test
    prompt_counts = [5, 10, 15, 20]

    # Ensure the directory exists
    os.makedirs('Evaluation/Interface', exist_ok=True)
    
    # Prepare the CSV file
    with open('Evaluation/Interface/stress_test_results.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["prompt_count", "average_response_time", "max_response_time", "min_response_time", "total_response_time"])
        
        for count in prompt_counts:
            print(f"Sending {count} prompts...")
            with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
                futures = [executor.submit(send_request, i) for i in range(count)]
                results = []
                response_times = []
                
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    results.append(result["response"])
                    response_times.append(result["response_time"])
                
                # Calculate metrics
                if response_times:
                    avg_response_time = sum(response_times) / len(response_times)
                    max_response_time = max(response_times)
                    min_response_time = min(response_times)
                    total_response_time = sum(response_times)
                else:
                    avg_response_time = max_response_time = min_response_time = total_response_time = 0
                
                # Write metrics to the CSV file
                writer.writerow([count, avg_response_time, max_response_time, min_response_time, total_response_time])
            
            print(f"Metrics for {count} prompts written to Evaluation/Interface/stress_test_results.csv")

if __name__ == "__main__":
    main()



