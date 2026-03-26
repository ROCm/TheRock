# S3 Index Lambda

AWS Lambda that regenerates `index.html` files in S3 when CI artifacts are
uploaded. Triggered via an SQS queue with a batch window so that many uploads
to the same directory collapse into a single index regeneration.

## S3 key structure

```
{run_prefix}/logs/...         -- build logs
{run_prefix}/manifests/...   -- manifests
{run_prefix}/python/...      -- python packages
{run_prefix}/...             -- any other subdirectory
```

where `run_prefix` is `[{external_repo}/]{run_id}-{platform}`,
e.g. `12345678901-linux` or `ROCm-TheRock/12345678901-linux`.

## Deployment package

The Lambda deployment package is a flat zip. Assemble it as follows:

```bash
BUILD=/tmp/lambda-build
rm -rf $BUILD && mkdir -p $BUILD

# Install boto3 and its dependencies into the package root.
pip install boto3 -t $BUILD

# Copy the handler and its dependencies from the TheRock repo.
cp build_tools/aws_lambda/s3_index_handler.py   $BUILD/
cp build_tools/generate_s3_index.py             $BUILD/
mkdir -p $BUILD/_therock_utils
cp build_tools/_therock_utils/storage_backend.py  $BUILD/_therock_utils/
cp build_tools/_therock_utils/storage_location.py $BUILD/_therock_utils/

# Create the zip.
cd $BUILD && zip -r /tmp/s3_index_handler.zip .

# Deploy.
aws lambda update-function-code \
  --function-name <your-function-name> \
  --zip-file fileb:///tmp/s3_index_handler.zip
```

Resulting zip layout:

```
s3_index_handler.py
generate_s3_index.py
_therock_utils/
    storage_backend.py
    storage_location.py
boto3/
botocore/
...
```

**Runtime:** Python 3.12+
**Handler:** `s3_index_handler.lambda_handler`
**Timeout:** 60 seconds (recommended)

## Infrastructure

### SQS queue

Create a Standard queue with:

- **Visibility timeout:** ≥ Lambda timeout (60 s recommended)
- **Dead-letter queue:** recommended for failed-message inspection

### S3 event notifications

Configure PutObject event notifications on both buckets to send to the queue:

- `therock-ci-artifacts`
- `therock-ci-artifacts-external`

### Lambda event source mapping

Add an SQS trigger on the Lambda with:

- **Batch size:** 100
- **Batch window:** 30 seconds

### IAM — Lambda execution role

Add to the Lambda execution role:

```json
{
  "Effect": "Allow",
  "Action": [
    "sqs:ReceiveMessage",
    "sqs:DeleteMessage",
    "sqs:GetQueueAttributes"
  ],
  "Resource": "arn:aws:sqs:<region>:<account-id>:<queue-name>"
}
```

The existing S3 permissions (`s3:GetObject`, `s3:PutObject`, `s3:ListBucket`)
on the target buckets remain unchanged.

### IAM — SQS queue resource policy

Allow S3 to send messages to the queue:

```json
{
  "Effect": "Allow",
  "Principal": {"Service": "s3.amazonaws.com"},
  "Action": "sqs:SendMessage",
  "Resource": "arn:aws:sqs:<region>:<account-id>:<queue-name>",
  "Condition": {
    "ArnLike": {
      "aws:SourceArn": [
        "arn:aws:s3:::therock-ci-artifacts",
        "arn:aws:s3:::therock-ci-artifacts-external"
      ]
    }
  }
}
```

The `Condition` scopes the permission to the two CI buckets only.
