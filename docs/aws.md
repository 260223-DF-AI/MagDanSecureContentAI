# AWS Set-Up guide

## Steps by doing it through an extension
### Step 1-
Install AWS Toolkit extension:

1. Open VS Code
2. Go to extensions
3. Type in the search: AWS Toolkit
4. Install the extension

### Step 2-
Configure AWS credentials

1. run in the terminal: aws configure
2. the console will prompt you for -
* Access key
* Secret key
* Region
* output format

3. YOU CAN do this same thing through the extension with "using IAM credentials"
- this is the 3rd option

### Step 3-
Connect to SageMaker

1. In the AWS Toolkit extension panel...
2. Expand SageMaker
    - you can see notebooks, training jobs, any end points


## Steps to get keys:
1. In the IAM console... go to users and click on user name
- if it's your user, though, click on your name in the upper right hand corner.
2. Click the tab "security credentials"
3. Scroll down to access keys -> create access key
- this will generate your access and secret key (jot this down somewhere)