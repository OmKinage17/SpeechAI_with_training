import os
import sys
import uuid
import hmac
import hashlib
from datetime import datetime, timezone

# Ensure UTF-8 output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Ensure backend root is in sys.path for test discovery
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from fastapi.testclient import TestClient
import main
from main import app, hash_password, verify_password, generate_token, verify_token, SECRET_KEY

class InMemoryUsersCollection:
    """In-memory collection mimicking MongoDB collection for isolated unit testing."""
    def __init__(self):
        self.docs = []

    def find_one(self, query):
        for doc in self.docs:
            if all(doc.get(k) == v for k, v in query.items()):
                return doc.copy()
        return None

    def insert_one(self, doc):
        self.docs.append(doc.copy())
        return type("InsertResult", (), {"inserted_id": doc.get("_id")})()

    def delete_many(self, query):
        initial = len(self.docs)
        self.docs = [d for d in self.docs if not all(d.get(k) == v for k, v in query.items())]
        return type("DeleteResult", (), {"deleted_count": initial - len(self.docs)})()

class InMemoryDatabase:
    """In-memory database dictionary mimicking PyMongo Database."""
    def __init__(self):
        self.collections = {"users": InMemoryUsersCollection()}

    def __getitem__(self, name):
        if name not in self.collections:
            self.collections[name] = InMemoryUsersCollection()
        return self.collections[name]

def test_password_hashing():
    print("Testing password hashing PBKDF2 helpers...")
    pwd = "MySecretPassword123"
    hashed = hash_password(pwd)
    print(f"Hashed format: {hashed}")
    
    assert verify_password(pwd, hashed) is True, "Password verification should pass"
    assert verify_password("WrongPassword", hashed) is False, "Password verification should fail with wrong password"
    assert verify_password(pwd, "invalid_format") is False, "Verification should fail on invalid hash format"
    print("[SUCCESS] Password hashing tests passed!")

def test_tokens():
    print("\nTesting HMAC token helpers...")
    user_id = "test-user-123"
    token = generate_token(user_id)
    print(f"Generated token: {token}")
    
    verified_id = verify_token(token)
    assert verified_id == user_id, f"Verified user ID should be {user_id}, got {verified_id}"
    
    # Test tampering
    tampered_token = token.replace("test-user-123", "tampered-user")
    assert verify_token(tampered_token) is None, "Verification should fail for tampered token"
    
    # Test malformed token
    assert verify_token("malformed:token") is None, "Verification should fail for malformed token"
    assert verify_token("") is None, "Verification should fail for empty token"
    
    # Test expired token (> 7 days)
    old_timestamp = str(int(datetime.now(timezone.utc).timestamp() - (8 * 86400)))
    msg = f"{user_id}:{old_timestamp}"
    old_sig = hmac.new(SECRET_KEY.encode('utf-8'), msg.encode('utf-8'), hashlib.sha256).hexdigest()
    expired_token = f"{user_id}:{old_timestamp}:{old_sig}"
    assert verify_token(expired_token) is None, "Verification should fail for expired token"
    print("[SUCCESS] Token helper tests passed!")

def test_endpoints():
    """Tests registration & login endpoints via FastAPI TestClient with in-memory isolation."""
    print("\nTesting registration & login endpoints via in-memory TestClient...")
    orig_db = main.db
    main.db = InMemoryDatabase()
    client = TestClient(app)
    
    try:
        test_id = str(uuid.uuid4())[:8]
        email = f"user_{test_id}@test.com"
        name = "Test User"
        password = "SuperSecretPassword"
        
        # 1. Register User
        reg_payload = {"name": name, "email": email, "password": password}
        response = client.post("/auth/register", json=reg_payload)
        print("Register Response:", response.status_code, response.json())
        assert response.status_code == 200, "Registration should succeed"
        reg_data = response.json()
        assert reg_data.get("status") == "success"

        # 2. Register Duplicate Email (should fail)
        response_dup = client.post("/auth/register", json=reg_payload)
        print("Register Duplicate Response:", response_dup.status_code, response_dup.json())
        assert response_dup.status_code == 400, "Duplicate registration should fail with 400"
        assert "Email already registered" in response_dup.json().get("detail", "")

        # 3. Login with Correct Password
        login_payload = {"email": email, "password": password}
        response_login = client.post("/auth/login", json=login_payload)
        print("Login Correct Response:", response_login.status_code)
        assert response_login.status_code == 200, "Login should succeed"
        login_data = response_login.json()
        assert "access_token" in login_data, "Login response must contain access_token"
        assert login_data.get("token_type") == "bearer"
        assert login_data.get("user", {}).get("email") == email
        assert login_data.get("user", {}).get("name") == name
        
        # Verify access token validity with token helper
        token_user = verify_token(login_data["access_token"])
        assert token_user == login_data["user"]["id"], "Returned token must verify to correct user ID"
        
        # 4. Login with Incorrect Password
        login_bad_payload = {"email": email, "password": "WrongPassword"}
        response_bad_login = client.post("/auth/login", json=login_bad_payload)
        print("Login Bad Response:", response_bad_login.status_code, response_bad_login.json())
        assert response_bad_login.status_code == 400, "Login with wrong password should fail with 400"

        # 5. Login with Non-Existent User
        login_ghost_payload = {"email": "nonexistent@test.com", "password": password}
        response_ghost = client.post("/auth/login", json=login_ghost_payload)
        assert response_ghost.status_code == 400, "Login with non-existent user should fail with 400"

        print("[SUCCESS] In-memory Register & Login endpoint tests passed!")
    finally:
        main.db = orig_db

def test_database_unavailable_error():
    """Tests 500 error contract when the database is unavailable."""
    print("\nTesting database unavailable error handling...")
    orig_db = main.db
    main.db = None
    client = TestClient(app)
    
    try:
        reg_resp = client.post("/auth/register", json={"name": "NoDB", "email": "nodb@test.com", "password": "pwd"})
        assert reg_resp.status_code == 500, "Register should fail with 500 when DB is unavailable"
        assert "Database not available" in reg_resp.json().get("detail", "")
        
        login_resp = client.post("/auth/login", json={"email": "nodb@test.com", "password": "pwd"})
        assert login_resp.status_code == 500, "Login should fail with 500 when DB is unavailable"
        assert "Database not available" in login_resp.json().get("detail", "")
        print("[SUCCESS] Database unavailable error handling tests passed!")
    finally:
        main.db = orig_db

def test_live_mongodb_integration():
    """Tests end-to-end registration and login against live MongoDB if running, with guaranteed teardown."""
    if main.db is None:
        print("\n[SKIP] Live MongoDB not connected; skipping live DB persistence test.")
        return
        
    print("\nTesting registration & login with Live MongoDB persistence & teardown...")
    client = TestClient(app)
    test_id = str(uuid.uuid4())[:8]
    email = f"live_test_{test_id}@speechai.test"
    password = "LiveSecretPassword123"
    name = "Live Test User"
    
    try:
        # Register
        r_reg = client.post("/auth/register", json={"name": name, "email": email, "password": password})
        assert r_reg.status_code == 200, "Live registration failed"
        
        # Verify document in DB
        db_doc = main.db["users"].find_one({"email": email})
        assert db_doc is not None, "User document must exist in MongoDB"
        assert db_doc.get("name") == name
        
        # Login
        r_login = client.post("/auth/login", json={"email": email, "password": password})
        assert r_login.status_code == 200, "Live login failed"
        assert "access_token" in r_login.json()
        print("[SUCCESS] Live MongoDB persistence and verification passed!")
    finally:
        # Teardown test record
        del_res = main.db["users"].delete_many({"email": email})
        print(f"Teardown: deleted {del_res.deleted_count} live test record(s) from MongoDB.")

if __name__ == "__main__":
    test_password_hashing()
    test_tokens()
    test_endpoints()
    test_database_unavailable_error()
    test_live_mongodb_integration()
    print("\nALL PORTABLE CRYPTOGRAPHIC AUTH TESTS PASSED SUCCESSFULLY!")
