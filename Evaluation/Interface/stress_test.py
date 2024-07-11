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
        end_time = time.time()
        response_time = end_time - start_time

        # Log response content for debugging
        print(f"Response content for prompt {prompt_num}: {response_data}")
        
        return {"response": response_data, "response_time": response_time, "start_time": start_time}
    except requests.exceptions.RequestException as e:
        end_time = time.time()
        response_time = end_time - start_time
        print(f"Request failed for prompt {prompt_num}: {e}")
        return {"response": {"error": str(e)}, "response_time": response_time, "start_time": start_time}

def check_task_status(task_id):
    """
    Checks the status of a given task until it is completed.
    
    Args:
        task_id (int): The task ID to check the status for.
        
    Returns:
        float: The total time taken for the task to complete.
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
        writer.writerow(["prompt_count", "average_response_time", "max_response_time", "min_response_time", "total_response_time"])
        
        for count in prompt_counts:
            print(f"Sending {count} prompts...")
            with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
                futures = [executor.submit(send_request, i) for i in range(count)]
                results = []
                response_times = []
                
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    results.append(result)
                    response_times.append(result["response_time"])

                # Wait for all tasks to complete
                task_completion_times = []
                for result in results:
                    if "taskID" in result["response"]:
                        task_id = result["response"]["taskID"]
                        completion_time = check_task_status(task_id)
                        total_time = completion_time - result["start_time"]
                        task_completion_times.append(total_time)

                # Calculate metrics
                if task_completion_times:
                    avg_response_time = sum(task_completion_times) / len(task_completion_times)
                    max_response_time = max(task_completion_times)
                    min_response_time = min(task_completion_times)
                    total_response_time = sum(task_completion_times)
                else:
                    avg_response_time = max_response_time = min_response_time = total_response_time = 0
                
                # Write metrics to the CSV file
                writer.writerow([count, avg_response_time, max_response_time, min_response_time, total_response_time])
            
            print(f"Metrics for {count} prompts written to Evaluation/Interface/stress_test_results.csv")

if __name__ == "__main__":
    main()


