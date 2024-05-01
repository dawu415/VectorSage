# Cloud Controller blobstore
Cloud Foundry uses a blobstore to store the source code that enables you to
push, stage, and run.
This topic references staging and treats all blobstores as generic object stores.
For more information about staging, see [How Apps Are Staged](https://docs.cloudfoundry.org/concepts/how-applications-are-staged.html).
For more information about how specific third-party blobstores can be configured, see [Cloud Controller Blobstore Configuration](https://docs.cloudfoundry.org/deploying/common/cc-blobstore-config.html).

## Staging using the blobstore
This section describes how staging buildpack apps uses the blobstore.
The following diagram illustrates how the staging process uses the blobstore.
To walk through the same diagram in an app staging context, see [How Diego Stages Buildpack Apps](https://docs.cloudfoundry.org/concepts/how-applications-are-staged.html).
The staging process for buildpack apps includes a developer and the following components: CF Command Line, Cloud Controller (CCNG), Blobstore, cc-uploader, Diego Cell (Staging), and Diego Cell (Running).

* Step 1: cf push from Developer to CF CLI.

* Step 2: Checksum source files from Developer to CF CLI.

* Step 3: Resource Match from CF CLI to the CCNG.

* Step 4: Check file existence from the CCNG to Blobstore.

* Step 5: Upload unmatched files from CF CLI to CCNG.

* Step 6: Download cached files from Blobstore to CCNG.

* Step 7: Upload complete package from CCNG to Blobstore.

* Step 8: Download package and buildpack from Blobstore to Diego Cell (Staging).

* Step 9: Upload droplet from Diego Cell (Staging) through cc-uploader, then CCNG, to the Blobstore.

* Step 10: Download droplet from Blobstore to CCNG.
![Full description of this diagram is in the text.](https://docs.cloudfoundry.org/concepts/images/cc-blobstore-staging.png)
The process in which the staging process uses the blobstore is as follows:

1. **cf push:** A developer runs `cf push`.

2. **Create app:** The Cloud Foundry Command Line Interface (cf CLI) gathers local source code files and computes a checksum of each.

3. **Store app metadata:** The cf CLI makes a `resource_matches` request, which matches resources to Cloud Controller. The request lists file names and their checksums. For more information and an example API request, see [Resource Matches](http://v3-apidocs.cloudfoundry.org/version/3.74.0/index.html#resource-matches) in the Cloud Foundry API documentation.

4. **Check file existence** includes the following:

1. The Cloud Controller makes a series of `HEAD` requests to the blobstore to find out which files it has cached.

2. Cloud Controller content-addresses its cached files so that changes to a file result in it being stored as a different object.

3. Cloud Controller computes which files it has and which it needs the cf CLI to upload. This process can take a long time.

- In response to the resource match request, Cloud Controller lists the files the cf CLI needs to upload.

5. **Upload unmatched files:** The cf CLI compresses and uploads the unmatched files to Cloud Controller.

6. **Download cached files:** Cloud Controller downloads, to its local disk, the matched files that are cached in the blobstore.

7. **Upload complete package** includes the following:

1. Cloud Controller compresses the newly uploaded files with the downloaded cached files in a ZIP file.

2. Cloud Controller uploads the complete package to the blobstore.

8. **Download package & buildpack(s):** A Diego Cell downloads the package and its buildpacks into a container and stages the app.

9. **Upload droplet** includes the following:

1. After the app has been staged, the Diego Cell uploads the complete droplet to `cc-uploader`.

2. `cc-uploader` makes a multi-part upload request to upload the droplet to Cloud Controller.

3. Cloud Controller enqueues an asynchronous job to upload to the blobstore.

10. **Download droplet** includes the following:

1. A Diego Cell attempts to download the droplet from Cloud Controller into the app container.

2. Cloud Controller asks the blobstore for a signed URL.

3. Cloud Controller redirects the Diego Cell droplet download request to the blobstore.

4. A Diego Cell downloads the app droplet from the blobstore and runs it.

## Blobstore load
The load that Cloud Controller generates on its blobstore is unique due to resource matching.
Many blobstores that perform well under normal read, write, and delete loada are overwhelmed by
Cloud Controller’s heavy use of HEAD requests during resource matching.
Pushing an app with large number of files causes Cloud Controller to check the blobstore for the existence of each file.
Parallel BOSH deployments of Diego Cells can also generate significant read load on the Cloud Controller blobstore as the cells perform evacuation. For more information, see the [Evacuation](https://docs.cloudfoundry.org/devguide/deploy-apps/app-lifecycle.html#evacuation) section of the *App Container Lifecycle* topic.

## How Cloud Controller reaps expired packages, droplets, and buildpacks
As new droplets and packages are created, the oldest ones associated with an app are marked as `EXPIRED` if they exceed the configured limits for packages and droplets stored per app.
Each night, starting at midnight, Cloud Controller runs a series of jobs to delete the data associated with expired packages, droplets, and buildpacks.
Enabling the native versioning feature on your blobstore increases the number of resources stored and costs. For more information, see [Using Versioning](https://docs.aws.amazon.com/AmazonS3/latest/dev/Versioning.html) in the AWS documentation.

## Blobstore interaction timeouts
Cloud Controller inherits default blobstore operation timeouts from Excon.
Excon defaults to 60 second read, write, and connect timeouts.
For more information, see the [excon](https://github.com/excon/excon) repository on GitHub.