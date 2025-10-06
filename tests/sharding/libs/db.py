import boto3
from . import utils
from .utils import log
import os

# Initialize DynamoDB client with error handling
dynamodb_table_name = "TestsCache"
try:
    dynamodb_client = boto3.client(
        "dynamodb", region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-2")
    )
    # Test if we can actually use DynamoDB
    dynamodb_client.describe_table(TableName=dynamodb_table_name)
    DYNAMODB_AVAILABLE = True
    log("DynamoDB client initialized successfully")
except Exception as e:
    log(f"DynamoDB not available: {e}")

    # Print all environment variables
    for key, value in os.environ.items():
        print(f"{key} = {value}")
    dynamodb_client = None
    DYNAMODB_AVAILABLE = False


def get_dynamodb_item(test_suite):
    """
    Retrieves an item from the TestsCache DynamoDB table using test_suite and test_name as keys
    Returns the item if found, None if not found
    """
    try:
        response = dynamodb_client.get_item(
            TableName=dynamodb_table_name,
            Key={"test_suite": {"S": test_suite}},
        )
        if "Item" in response:
            log(f"Retrieved cache data for test_suite: {test_suite}")
            return response["Item"]
        else:
            log(f"No cache data found for test_suite: {test_suite}")
            return dict()
    except Exception as e:
        log(f"Error retrieving cache data from DynamoDB: {str(e)}")
        return dict()


def put_dynamodb_item(test_suite, item_data):
    """
    Stores an item in the TestsCache DynamoDB table with test_suite as key
    item_data should be a dictionary with DynamoDB attribute format
    Returns True if successful, False otherwise
    """
    try:
        # Ensure required keys are present
        item = {"test_suite": {"S": test_suite}}

        # Add additional data from item_data
        if item_data:
            item.update(item_data)

        dynamodb_client.put_item(TableName=dynamodb_table_name, Item=item)
        log(f"Successfully stored cache data for test_suite: {test_suite}")
        return True
    except Exception as e:
        log(f"Error storing cache data to DynamoDB: {str(e)}")
        return False


def get_all_dynamodb_items():
    """
    Retrieves all items from the TestsCache DynamoDB table
    Returns a list of all items if successful, empty list if none found or on error
    """
    try:
        all_items = []
        response = dynamodb_client.scan(TableName=dynamodb_table_name)

        # Add items from first page
        if "Items" in response:
            all_items.extend(response["Items"])

        # Handle pagination if there are more items
        while "LastEvaluatedKey" in response:
            response = dynamodb_client.scan(
                TableName=dynamodb_table_name,
                ExclusiveStartKey=response["LastEvaluatedKey"],
            )
            if "Items" in response:
                all_items.extend(response["Items"])

        log(
            f"Retrieved {len(all_items)} items from DynamoDB table {dynamodb_table_name}"
        )
        return all_items
    except Exception as e:
        log(f"Error retrieving all items from DynamoDB: {str(e)}")
        return []
