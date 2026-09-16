import requests
import sys

API_URL = "http://127.0.0.1:8000"

def test_pipeline():
    # 1. Login to get token
    print("\n--- Logging in as exec_user ---")
    login_url = "http://127.0.0.1:8000/login"
    login_data = {
        "username": "exec_user",
        "password": "exec_user"
    }
    response = requests.post(login_url, data=login_data)

    if response.status_code != 200:
        print("Login failed:", response.text)
        return

    token = response.json()["access_token"]
    print("Success! Got JWT Token.")

    print(f"\n--- Sending query ---")
    print(f"Query: {query}")
    chat_response = requests.post(
        f"{API_URL}/chat",
        headers={"Authorization": f"Bearer {token}"},
        json={"query": query}
    )

    if chat_response.status_code != 200:
        print("Chat failed:", chat_response.text)
        return

    print("\n--- AI Response ---")
    print(chat_response.json()["response"])
    print("-" * 50)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = sys.argv[1]
    else:
        query = "What does the assessment evaluate regarding analytical maturity?"

    # You can change the user below to test different role restrictions!
    # Available users: tech_user, exec_user, ops_user
    run_query("exec_user", "exec_user", query)
