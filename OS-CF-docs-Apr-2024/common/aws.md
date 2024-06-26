# Deploying BOSH on AWS
This topic describes how to use the [bosh-bootloader](https://github.com/cloudfoundry/bosh-bootloader) command-line tool to set up an environment for Cloud Foundry on Amazon Web Services (AWS) and deploy a [BOSH Director](https://bosh.io/docs/bosh-components.html#director).

## Overview
This topic describes how to create:

1. A BOSH Director instance

2. A bastion instance

3. A set of randomly generated BOSH Director credentials

4. A generated key pair that allows you to SSH into the BOSH Director and any instances that BOSH deploys

5. A copy of the manifest used to deploy the BOSH Director

**Note:** A manifest is a YAML file that defines the components and properties of a BOSH deployment. For more information, see [Deployment Manifest](https://bosh.io/docs/deployment-manifest.html) in the BOSH documentation.

6. A basic cloud config

**Note:** A cloud config is a YAML file that defines IaaS-specific configuration for BOSH. For more information, see [Usage](https://bosh.io/docs/cloud-config.html) in the BOSH documentation.

7. A set of Elastic Load Balancers (ELBs)

**Note:** bosh-bootloader creates the ELBs, but you must still configure DNS to point your domains to the ELBs. For more information, see [Setting Up DNS for Your Environment](https://docs.cloudfoundry.org/deploying/common/dns_prereqs.html).

## Step 1: Download Dependencies
To download the required dependencies for bosh-bootloader:

1. Download [Terraform](https://www.terraform.io/downloads.html) v0.9.1 or later. Unzip the file and move it to somewhere in your PATH:
```
$ tar xvf ~/Downloads/terraform*
$ sudo mv ~/Downloads/terraform /usr/local/bin/terraform
```

2. Download [BOSH CLI v2+](https://bosh.io/docs/cli-v2.html#install). Make the binary executable and move it to somewhere in your PATH:
```
$ chmod +x ~/Downloads/bosh-cli-*
$ sudo mv ~/Downloads/bosh-cli-* /usr/local/bin/bosh
```

3. To download and install bosh-bootloader, do one of the following:

* On Mac OS X, use Homebrew:
```
$ brew install cloudfoundry/tap/bbl
```

* Download the latest bosh-bootloader from [GitHub](https://github.com/cloudfoundry/bosh-bootloader/releases/latest). Make the binary executable and move it to somewhere in your PATH:
```
$ chmod +x ~/Downloads/bbl-*
$ sudo mv ~/Downloads/bbl-* /usr/local/bin/bbl
```

4. Install the [AWS CLI](https://aws.amazon.com/cli/).

## Step 2: Create an IAM User
To create the Identity and Access Management (IAM) user that bosh-bootloader needs to interact with AWS:

1. Configure the AWS CLI with the information and credentials from your AWS account by running:
```
aws configure
AWS Access Key ID [None]: YOUR-AWS-ACCESS-KEY-ID
AWS Secret Access Key [None]: YOUR-AWS-SECRET-ACCESS-KEY
Default region name [None]: YOUR-AWS-REGION
Default output format [None]: json
```
Where:

* `YOUR-AWS-ACCESS-KEY-ID` is your AWS access key ID.

* `YOUR-AWS-SECRET-ACCESS-KEY` is your AWS secret access key.

* `YOUR-AWS-REGION` is the AWS region whose servers you want to send your requests to by default.
For more information about retrieving your credentials, see [Configuring the AWS CLI](http://docs.aws.amazon.com/cli/latest/userguide/cli-chap-getting-started.html) in the AWS documentation.

2. Create the IAM user for bosh-bootloader with the AWS CLI by running:
```
aws iam create-user --user-name "bbl-user"
```

3. Copy the following policy text to your clipboard:
```
{
"Version": "2012-10-17",
"Statement": [
{
"Effect": "Allow",
"Action": [
"ec2:\*",
"cloudformation:\*",
"elasticloadbalancing:\*",
"iam:\*",
"route53:\*",
"logs:\*",
"kms:\*"
],
"Resource": [
"\*"
]
}
]
}
```

4. Apply the policy by running:
```
aws iam put-user-policy --user-name "bbl-user" \

--policy-name "bbl-policy" \

--policy-document "$(pbpaste)"
```

5. Create an access key by running:
```
aws iam create-access-key --user-name "bbl-user"
```
This command outputs an `Access Key ID` and a `Secret Access Key`. Record these values and store them in a secure place. You use them in the next section.

## Step 3: Create Infrastructure, Bastion, BOSH Director, and Load Balancers
To create the required infrastructure and deploy a BOSH Director, run:
```
bbl plan \

--name YOUR-ENV-NAME \

--iaas aws \

--aws-access-key-id YOUR-ACCESS-KEY-ID \

--aws-secret-access-key YOUR-SECRET-ACCESS-KEY \

--aws-region YOUR-AWS-REGION \

--lb-type cf \

--lb-cert YOUR-CERT.crt \

--lb-key YOUR-KEY.key \

--lb-domain YOUR-ENV-NAME.YOUR-SYSTEM-DOMAIN<br/>
bbl up
```
Where:

* `YOUR-ACCESS-KEY-ID` and `YOUR-SECRET-ACCESS-KEY` are the credentials for the `bbl-user` you created in the previous section.

* `YOUR-AWS-REGION` is your AWS region, such as `us-west-2`.
The `bbl up` command takes five to eight minutes to complete.
After `bbl` deploys the BOSH Director, you must point `YOUR-SYSTEM-DOMAIN` at the BOSH Director’s name servers. For example, if you are using AWS Route53 to manage `YOUR-SYSTEM-DOMAIN`:

1. Run:
```

--lb-domain YOUR-ENV-NAME.YOUR-SYSTEM-DOMAIN
```

2. See the list of name servers for the BOSH Director by running:
```
bbl outputs | yq .env_dns_zone_name_servers
```

3. Log in to the AWS Route 53 dashboard and go to **Registered Domains**.

4. Choose `YOUR-SYSTEM-DOMAIN`.

5. Click **Add/Edit Name Servers**.

6. Add a `YOUR-ENV-NAME.YOUR-SYSTEM-DOMAIN` NS record, and add the name servers found in the output of the above `bbl outputs` command to that record.
For more information, see [Adding or Changing Name Servers or Glue Records](https://docs.aws.amazon.com/Route53/latest/DeveloperGuide/domain-name-servers-glue-records.html#domain-name-servers-glue-records-procedure) in the AWS documentation.
When `bbl plan` or `bbl up` is run, files in the `--state-dir` (or present working directory) will be created, modified, or deleted.

**Note:** The **bbl state directory** contains credentials and other metadata related to your BOSH Director and infrastructure. Back up this directory and store it in a safe location.
To extract information from the bbl state, use `bbl`. For example, to obtain your BOSH Director address, run:
```
bbl director-address
```
Run `bbl` to see the full list of values from the state file that you can print. You must always run `bbl` from the state directory.
For more information about the options for securing HTTP traffic into your Cloud Foundry deployment with TLS certificates, see [Securing Traffic into Cloud Foundry](https://docs.cloudfoundry.org/adminguide/securing-traffic.html).
For test and development environments, you can also generate your own CA certificate and key with a tool such as [certstrap](https://github.com/square/certstrap).

## Step 4: Connect to the BOSH Director
To connect to the BOSH Director, run:
```
eval "$(bbl print-env)"
```

## Destroy the BOSH Resources
You can use `bbl destroy` to delete the BOSH Director infrastructure in your AWS environment. Use this command if `bbl up` does not complete successfully and you want to reset your environment, or if you want to destroy the resources created by bosh-bootloader for any other reason.
To delete load balancers only, run:
```
bbl plan
bbl up
```
To delete the infrastructure, bastion, director, and load balancers, run:
```
bbl destroy
```