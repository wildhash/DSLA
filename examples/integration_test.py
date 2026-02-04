#!/usr/bin/env python3
"""
Comprehensive integration test for DSLA API server.
Tests all endpoints with real data.
"""

import requests
import json
import sys


BASE_URL = "http://localhost:8000"


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def test_root():
    """Test the root endpoint."""
    print_section("Testing Root Endpoint")
    response = requests.get(f"{BASE_URL}/")
    print(f"Status Code: {response.status_code}")
    print(f"Response:\n{json.dumps(response.json(), indent=2)}")
    return response.status_code == 200


def test_list_adapters():
    """Test listing adapters."""
    print_section("Testing List Adapters")
    response = requests.get(f"{BASE_URL}/adapters")
    print(f"Status Code: {response.status_code}")
    data = response.json()
    print(f"Found {len(data['adapters'])} adapters:")
    for adapter in data['adapters']:
        print(f"  - {adapter['domain']}: {adapter['keywords']}")
    return response.status_code == 200


def test_adapt_legal():
    """Test /adapt endpoint with legal document."""
    print_section("Testing /adapt with Legal Document")
    
    payload = {
        "domain": "legal_doc",
        "input_data": {
            "document_type": "employment_contract",
            "content": "This is a sample employment contract..."
        }
    }
    
    response = requests.post(f"{BASE_URL}/adapt", json=payload)
    print(f"Status Code: {response.status_code}")
    data = response.json()
    print(f"Domain: {data['domain']}")
    print(f"Adapted Input: {json.dumps(data['adapted_input'], indent=2)}")
    print(f"Number of tools: {len(data['tools'])}")
    print(f"Prompt preview: {data['prompt'][:200]}...")
    return response.status_code == 200


def test_adapt_trading():
    """Test /adapt endpoint with trading data."""
    print_section("Testing /adapt with Trading Data")
    
    payload = {
        "domain": "trading_ops",
        "input_data": {
            "asset": "BTC/USD",
            "market_data": {
                "price": 45000,
                "volume": 1234567,
                "rsi": 62
            }
        }
    }
    
    response = requests.post(f"{BASE_URL}/adapt", json=payload)
    print(f"Status Code: {response.status_code}")
    data = response.json()
    print(f"Domain: {data['domain']}")
    print(f"Adapted Input: {json.dumps(data['adapted_input'], indent=2)}")
    print(f"Number of tools: {len(data['tools'])}")
    return response.status_code == 200


def test_routing():
    """Test automatic routing based on query."""
    print_section("Testing Query-Based Routing")
    
    test_queries = [
        ("I need to analyze a legal contract", "legal_doc"),
        ("What's the market trend for Bitcoin?", "trading_ops"),
    ]
    
    all_passed = True
    for query, expected_domain in test_queries:
        payload = {
            "query": query,
            "input_data": {
                "document_type": "contract" if "legal" in query else None,
                "content": "sample" if "legal" in query else None,
                "asset": "BTC/USD" if "Bitcoin" in query else None,
                "market_data": {"price": 45000} if "Bitcoin" in query else None,
            }
        }
        # Remove None values
        payload["input_data"] = {k: v for k, v in payload["input_data"].items() if v is not None}
        
        response = requests.post(f"{BASE_URL}/adapt", json=payload)
        data = response.json()
        matched = data['domain'] == expected_domain
        all_passed = all_passed and matched
        print(f"Query: '{query}'")
        print(f"  → Routed to: {data['domain']}")
        print(f"  → Expected: {expected_domain}")
        print(f"  → Match: {'✓' if matched else '✗'}")
        print()
    
    return all_passed


def test_run_workflow():
    """Test the full /run workflow."""
    print_section("Testing /run Workflow")
    
    payload = {
        "domain": "trading_ops",
        "input_data": {
            "asset": "ETH/USD",
            "market_data": {"price": 2500, "volume": 987654}
        },
        "save_to_memory": True
    }
    
    response = requests.post(f"{BASE_URL}/run", json=payload)
    print(f"Status Code: {response.status_code}")
    data = response.json()
    print(f"Domain: {data['domain']}")
    print(f"Output Summary: {data['adapted_output']['summary']}")
    print(f"Memory ID: {data['memory_id']}")
    return response.status_code == 200 and data['memory_id']


def test_memory_operations():
    """Test memory store and retrieve."""
    print_section("Testing Memory Operations")
    
    # Store
    payload = {
        "domain": "test",
        "key": "integration_test_key",
        "value": {"test": "data", "timestamp": "2024-01-01"},
        "metadata": {"source": "integration_test"}
    }
    
    response = requests.post(f"{BASE_URL}/memory/store", json=payload)
    print(f"Store Status: {response.status_code}")
    memory_id = response.json()['id']
    print(f"Stored with ID: {memory_id}")
    
    # Retrieve
    response = requests.get(f"{BASE_URL}/memory/test/integration_test_key")
    print(f"Retrieve Status: {response.status_code}")
    data = response.json()
    print(f"Retrieved: {json.dumps(data['value'], indent=2)}")
    
    # Query domain
    response = requests.get(f"{BASE_URL}/memory/test")
    print(f"Query Status: {response.status_code}")
    entries = response.json()['entries']
    print(f"Found {len(entries)} entries in 'test' domain")
    
    return response.status_code == 200


def main():
    """Run all tests."""
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 15 + "DSLA API Integration Test Suite" + " " * 31 + "║")
    print("╚" + "═" * 78 + "╝")
    
    tests = [
        ("Root Endpoint", test_root),
        ("List Adapters", test_list_adapters),
        ("Adapt Legal Document", test_adapt_legal),
        ("Adapt Trading Data", test_adapt_trading),
        ("Query-Based Routing", test_routing),
        ("Run Full Workflow", test_run_workflow),
        ("Memory Operations", test_memory_operations),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Error in {name}: {str(e)}")
            results.append((name, False))
    
    # Summary
    print_section("Test Summary")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}  {name}")
    
    print(f"\n  Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n  🎉 All tests passed! 🎉")
        sys.exit(0)
    else:
        print("\n  ⚠️  Some tests failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
