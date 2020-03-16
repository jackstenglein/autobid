import boto3
from botocore.exceptions import NoCredentialsError

ACCESS_KEY_INDEX = 2
SECRET_KEY_INDEX = 3

""" Reads AWS credentials from local file paper-scraper-credentials.csv """
def get_aws_credentials():
    with open('paper-scraper-credentials.csv', 'r') as f:
        f.readline()
        credentials = f.readline().split(',')
        return (credentials[ACCESS_KEY_INDEX], credentials[SECRET_KEY_INDEX])

""" Downloads a file from the given S3 bucket and saves it at the local filename """
def download_from_aws(filename, bucket, s3_filename):
    (access_key, secret_key) = get_aws_credentials()
    s3 = boto3.client('s3', aws_access_key_id=access_key, aws_secret_access_key=secret_key)

    try:
        s3.download_file(bucket, s3_filename, filename)
        return True
    except FileNotFoundError:
        print("The file was not found")
        return False
    except NoCredentialsError:
        print("Credentials not available")
        return False
    except:
        print("Unexpected error:", sys.exc_info()[0])
        return False

if (__name__=="__main__"):
    download_from_aws("test.pdf", "cs380s-security-project", "reviewer_papers/--M1XEkAAAAJ/--M1XEkAAAAJ:5nxA0vEk-isC.pdf")
