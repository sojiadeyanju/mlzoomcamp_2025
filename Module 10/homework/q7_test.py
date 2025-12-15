import requests
import json
import time
from threading import Thread

# --- Configuration ---
URL = "http://localhost:9696/predict"
CLIENT_DATA = {"job": "management", "duration": 400, "poutcome": "success"}
NUM_THREADS = 10  # Number of simultaneous connections to open
REQUESTS_PER_THREAD = 1000 # Number of requests each thread will send

def send_requests(thread_id):
    """Function to send a burst of requests to the endpoint."""
    session = requests.Session()
    success_count = 0
    
    print(f"Thread {thread_id}: Starting {REQUESTS_PER_THREAD} requests...")

    for i in range(REQUESTS_PER_THREAD):
        try:
            # We don't sleep here; we want to flood the server quickly
            response = session.post(URL, json=CLIENT_DATA, timeout=5)
            
            # Check if the request was successful
            if response.status_code == 200:
                success_count += 1
            else:
                print(f"Thread {thread_id}: Request failed with status {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            print(f"Thread {thread_id}: Connection error! Server or port-forwarding might be down.")
            break
        except requests.exceptions.Timeout:
            print(f"Thread {thread_id}: Request timed out.")
            
    print(f"Thread {thread_id}: Finished. Successful requests: {success_count}/{REQUESTS_PER_THREAD}")


if __name__ == "__main__":
    start_time = time.time()
    
    # Create and start all threads
    threads = []
    for i in range(NUM_THREADS):
        thread = Thread(target=send_requests, args=(i + 1,))
        threads.append(thread)
        thread.start()

    # Wait for all threads to finish
    for thread in threads:
        thread.join()

    end_time = time.time()
    total_requests = NUM_THREADS * REQUESTS_PER_THREAD
    
    print("\n--- Load Test Summary ---")
    print(f"Total Requests Sent: {total_requests}")
    print(f"Total Time: {end_time - start_time:.2f} seconds")
    print(f"Requests Per Second (RPS): {total_requests / (end_time - start_time):.2f}")