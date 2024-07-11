import requests
import concurrent.futures
import csv
import time
import os

# URL for the submit and status endpoints
submit_url = "http://127.0.0.1:5000/submit"
status_url = "http://127.0.0.1:5000/status"

# Sample data to be sent in the requests
data_template = {
    "prompt": "Floor plan of a small apartment, few rooms, one bathroom, big kitchen, many windows",
    "num_images": 1,
    "scheduler": "DDIM",
    "inference_steps": 50
}

def send_request(prompt_num):
    """
    Sends a POST request to the Flask server and measures the response time.
    
    Args:
        prompt_num (int): The number to append to the prompt for uniqueness.
        
    Returns:
        dict: A dictionary containing the JSON response from the server and the response time.
    """
    data = data_template.copy()
    data["prompt"] = f"Floor plan of a small apartment {prompt_num}"

    start_time = time.time()
    try:
        response = requests.post(submit_url, json=data)
        response.raise_for_status()  # Raise an error for bad status codes
        response_data = response.json()
        submission_time = time.time() - start_time

        # Log response content for debugging
        print(f"Response content for prompt {prompt_num}: {response_data}")
        
        return {"response": response_data, "submission_time": submission_time, "start_time": start_time}
    except requests.exceptions.RequestException as e:
        submission_time = time.time() - start_time
        print(f"Request failed for prompt {prompt_num}: {e}")
        return {"response": {"error": str(e)}, "submission_time": submission_time, "start_time": start_time}

def check_task_status(task_id):
    """
    Checks the status of a given task until it is completed.
    
    Args:
        task_id (int): The task ID to check the status for.
        
    Returns:
        float: The time when the task is completed.
    """
    while True:
        response = requests.get(f"{status_url}/{task_id}")
        if response.status_code == 200:
            status_data = response.json()
            if status_data["status"] == "done":
                return time.time()
        time.sleep(1)

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
        writer.writerow(["prompt_count", "average_submission_time", "max_submission_time", "min_submission_time", 
                         "average_processing_time", "max_processing_time", "min_processing_time",
                         "average_completion_time", "max_completion_time", "min_completion_time",
                         "total_completion_time", "error_rate", "throughput"])
        
        for count in prompt_counts:
            print(f"Sending {count} prompts...")
            with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
                futures = [executor.submit(send_request, i) for i in range(count)]
                results = []
                submission_times = []
                
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    results.append(result)
                    submission_times.append(result["submission_time"])

                # Wait for all tasks to complete
                processing_times = []
                completion_times = []
                errors = 0
                for result in results:
                    if "taskID" in result["response"]:
                        task_id = result["response"]["taskID"]
                        completion_time = check_task_status(task_id)
                        total_time = completion_time - result["start_time"]
                        processing_time = total_time - result["submission_time"]
                        processing_times.append(processing_time)
                        completion_times.append(total_time)
                    else:
                        errors += 1

                # Calculate metrics
                if submission_times:
                    avg_submission_time = sum(submission_times) / len(submission_times)
                    max_submission_time = max(submission_times)
                    min_submission_time = min(submission_times)
                else:
                    avg_submission_time = max_submission_time = min_submission_time = 0

                if processing_times:
                    avg_processing_time = sum(processing_times) / len(processing_times)
                    max_processing_time = max(processing_times)
                    min_processing_time = min(processing_times)
                else:
                    avg_processing_time = max_processing_time = min_processing_time = 0

                if completion_times:
                    avg_completion_time = sum(completion_times) / len(completion_times)
                    max_completion_time = max(completion_times)
                    min_completion_time = min(completion_times)
                    total_completion_time = sum(completion_times)
                    throughput = count / total_completion_time if total_completion_time > 0 else 0
                else:
                    avg_completion_time = max_completion_time = min_completion_time = total_completion_time = throughput = 0

                error_rate = errors / count if count > 0 else 0

                # Write metrics to the CSV file
                writer.writerow([count, avg_submission_time, max_submission_time, min_submission_time, 
                                 avg_processing_time, max_processing_time, min_processing_time,
                                 avg_completion_time, max_completion_time, min_completion_time,
                                 total_completion_time, error_rate, throughput])
            
            print(f"Metrics for {count} prompts written to Evaluation/Interface/stress_test_results.csv")

if __name__ == "__main__":
    main()



