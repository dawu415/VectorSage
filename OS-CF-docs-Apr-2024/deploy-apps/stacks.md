# Changing stacks
You can restage apps on a new stack. Here is a description of stacks and lists of stacks that are supported on Cloud Foundry v4-0.
To restage a Windows app on a new Windows stack, see [Changing Windows stacks](https://docs.cloudfoundry.org/devguide/deploy-apps/windows-stacks.html).
You can also use the Stack Auditor plug-in for the Cloud Foundry Command Line Interface (cf CLI) when changing stacks. See [Using the Stack Auditor plug-in](https://docs.cloudfoundry.org/adminguide/stack-auditor.html).

## Overview
A stack is a prebuilt root file system (rootfs) that supports a specific operating system. For example, Linux-based systems need `/usr` and `/bin` directories at their root. The stack works in tandem with a buildpack to support apps running in compartments. Under Diego architecture, cell VMs can support multiple stacks.

**Note**
Docker apps do not use stacks.

## Available stacks
Cloud Foundry v4-0 includes support for `cflinuxfs3`. The Linux `cflinuxfs3` stack is derived from Ubuntu Bionic 18.04. For more information about supported libraries, see the [GitHub stacks page](https://github.com/cloudfoundry/cflinuxfs3/blob/main/receipt.cflinuxfs3.x86_64).
The latest versions of Cloud Foundry include support for `cflinuxfs4` which is derived from Ubuntu 22.04 LTS (Jammy Jellyfish). For more information, see [GitHub cflinuxfs4 stack receipt](https://github.com/cloudfoundry/cflinuxfs4/blob/main/receipt.cflinuxfs4.x86_64).
You can also build your own custom stack. For more information, see [Adding a Custom Stack](https://docs.cloudfoundry.org/running/custom-stack.html).

## Restaging apps on a new stack
For security, stacks receive regular updates to address Common Vulnerabilities and Exposures ([CVEs](http://www.ubuntu.com/usn/)). Apps pick up on these stack changes through new releases of Cloud Foundry. However, if your app links statically to a library provided in the rootfs, you might have to manually restage it to pick up the changes.
It can be difficult to know what libraries an app statically links to, and it depends on the languages you are using. One example is an app that uses a Ruby or Python binary, and links out to part of the C standard library. If the C library requires an update, you might need to recompile the app and restage it.
To restage an app on a new stack:

1. Use the `cf stacks` command to list the stacks available in a deployment.
```
$ cf stacks
Getting stacks in org MY-ORG / space development as developer@example.com...
OK
name description
cflinuxfs3 Cloud Foundry Linux-based filesystem (Ubuntu 18.04)
cflinuxfs4 Cloud Foundry Linux-based filesystem (Ubuntu 22.04)
```

2. To change your stack and restage your app, run:
```
cf push MY-APP -s STACK-NAME
```
Where:

* MY-APP is the name of the app.

* STACK-NAME is the name of the new stack.For example, to restage your app on the stack `cflinuxfs4`, run `cf push MY-APP -s cflinuxfs4`:
```
$ cf push MY-APP -s cflinuxfs4
Using stack cflinuxfs4...
OK
Creating app MY-APP in org MY-ORG / space development as developer@example.com...
OK
...
requested state: started
instances: 1/1
usage: 1G x 1 instances
urls: MY-APP.cfapps.io
last uploaded: Wed Apr 8 23:40:57 UTC 2015
state since cpu memory disk

#0 running 2015-04-08 04:41:54 PM 0.0% 57.3M of 1G 128.8M of 1G
```

## Stacks API
For API information, see the *Stacks* section of the [Cloud Foundry API Documentation](https://v3-apidocs.cloudfoundry.org/).