# Upgrading dependency versions for Cloud Foundry
This topic tells you how to upgrade a dependency version in a custom buildpack.
These procedures enable Cloud Foundry (CF) operators to maintain custom buildpacks that contain
dependencies outside of the dependencies in the CF system buildpacks.

## Cloud Foundry buildpacks team process
The procedures in this topic refer to the tools used by the CF buildpacks team, but they do not require the following specific tools. You can use any continuous integration (CI) system and workflow management tool to update dependencies in custom buildpacks.
The CF buildpacks team uses the following tools to update dependencies:

* Concourse deployment of the [buildpacks-ci](https://github.com/cloudfoundry/buildpacks-ci) pipelines.

* [public-buildpacks-ci-robots](https://github.com/cloudfoundry/public-buildpacks-ci-robots) GitHub repository.

* [Pivotal Tracker](https://www.pivotaltracker.com) for workflow management.
When the [New Releases](https://buildpacks.ci.cf-app.com/pipelines/notifications/jobs/New%20Releases) job in the [notifications pipeline](https://buildpacks.ci.cf-app.com/pipelines/notifications)
detects a new version of a tracked dependency in a buildpack, it creates a [Tracker](https://www.pivotaltracker.com) story about building and including the new version of the dependency in the buildpack manifests. It also posts a message as the `dependency-notifier` to the [#buildpacks channel](https://cloudfoundry.slack.com/messages/buildpacks/) in the Cloud Foundry Slack channel.

## Building binaries
For all dependencies, you must build the binary from source or acquire the binary as a tarball from a trusted source. For most dependencies, the CF buildpacks team builds the binaries from source.
The following steps assume you are using a Concourse deployment of the `buildpacks-ci` pipelines and Pivotal Tracker.
To build the binary for a dependency:

1. Go to the `public-buildpacks-ci-robots` directory and verify no uncommitted changes exist.
```
$ cd ~/workspace/public-buildpacks-ci-robots
$ git status
```

1. Run the `git pull` command in the directory to ensure it contains the most recent version of the contents.
```
$ git pull -r
```

1. Go to the `binary-builds` directory.
```
$ cd binary-builds
```

1. Locate the YAML file for the buildpack that you want to build a binary. The directory contains YAML files for all the packages and dependencies tracked by the CF buildpacks team. Each YAML file correlates to the build queue for one dependency or package, and uses the naming format `DEPENDENCY-NAME.yml`. For example, the YAML file tracking the build queue for Ruby is named `ruby-builds.yml` and contains the following contents:
```

---
ruby: []
```

2. Different buildpacks use different signatures for verification. Determine which signature your buildpack requires by consulting the list in the [buildpacks](https://docs.cloudfoundry.org/buildpacks/upgrading_dependency_versions.html#buildpacks) section of this topic and follow the instructions to locate the SHA256, MD5, or GPG signature for the binary:

* For the SHA256 of a file, run `shasum -a 256 FILE-NAME`.

* For the MD5 of a file, run `md5 FILE-NAME`.

* For the GPG signature (for Nginx), see the [Nginx Downloads](http://nginx.org/en/download.html) page.

1. Add the version and verification for the new binary to the YAML file as attributes of an element under the dependency name. For example, to build the Ruby 2.3.0 binary verified with SHA256, add the following to the YAML file:
```

---
ruby:

- version: 2.3.0
sha256: ba5ba60e5f1aa21b4ef8e9bf35b9ddb57286cb546aac4b5a28c71f459467e507
```

**Important**
Do not preface the version number with the name of the binary or
language. For example, specify `2.3.0` for `version` instead of
`ruby-2.3.0`.
You can enqueue builds for multiple versions at the same time.
For example, to build both the Ruby 2.3.0 binary and the Ruby 2.3.1 binary, add the following to the YAML file:
```

---
ruby:

- version: 2.3.0
sha256: ba5ba60e5f1aa21b4ef8e9bf35b9ddb57286cb546aac4b5a28c71f459467e507

- version: 2.3.1
sha256: b87c738cb2032bf4920fef8e3864dc5cf8eae9d89d8d523ce0236945c5797dcd
```

2. Use the `git add` command to stage your changes:
```
$ git add .
```

3. Use the `git commit -m "YOUR-COMMIT-MESSAGE [#STORY-NUMBER]"` command to commit your changes using the Tracker story number. Replace `YOUR-COMMIT-MESSAGE` with your commit message, and `STORY-NUMBER` with the number of your Tracker story.
```
$ git commit -m "make that change [#1234567890]"
```

4. Run `git push` to push your changes to the remote origin.
```
$ git push
```

5. Pushing your changes initiates the binary building process, which you can monitor at the [binary-builder pipeline](https://buildpacks.ci.cf-app.com/pipelines/binary-builder)
of your own `buildpacks-ci` Concourse deployment. When the build completes, it adds a link to the Concourse build run to the Tracker story specified in the commit message for the new release.
Binary builds are run by the Cloud Foundry [Binary Builder](https://github.com/cloudfoundry/binary-builder) and the [dependency-builds](https://github.com/cloudfoundry/buildpacks-ci/blob/master/pipelines/dependency-builds.yml.erb) pipeline.

## Updating buildpack manifests
After you build the binary for a dependency that you want to access and download from a URL, follow the instructions to add the dependency version to the buildpack manifest
The following steps assume you are using a Concourse deployment of the `buildpacks-ci`
pipelines and Pivotal Tracker.
To add the dependency version to the buildpack manifest, use these steps:

1. Go to the directory of the buildpack for which you want to update dependencies and run `git checkout develop` to check out the `develop` branch.
```
$ cd ~/workspace/ruby-buildpack
$ git checkout develop
```

2. Edit the `manifest.yml` file for the buildpack to add or remove dependencies.
```
dependencies:

- name: ruby
version: 2.3.0
md5: 535342030a11abeb11497824bf642bf2
uri: https://pivotal-buildpacks.s3.amazonaws.com/concourse-binaries/ruby/ruby-2.3.0-linux-x64.tgz
cf\_stacks:

- cflinuxfs4
```
```

* Follow the current structure of the manifest. For example, if the manifest includes the two most recent patch versions for each minor version of the language, you can also include the two most recent patch versions for each minor version of the language, such as both `ruby-2.1.9` and `ruby-2.1.8`.

* Copy the `uri` and the `md5` from the `build-BINARY-NAME` job that ran in the Concourse dependency-builds pipeline and add them to the manifest.
In the PHP buildpack, you might see a <code>modules</code> line for each PHP dependency in
the manifest. Do not include this <code>modules</code> line in your new PHP dependency
entry. The <code>modules</code> line is added to the manifest by the <code>ensure-manifest-has-modules</code> Concourse job in the <code>php-buildpack</code> when you commit and push your changes. You can see this in the output logs of the <code>build-out</code> task.
```

1. Replace any other mentions of the old version number in the buildpack repository with the new version number. The CF buildpack team uses [Ag](https://github.com/ggreer/the_silver_searcher) for text searching.
```
$ ag OLD-VERSION
```

2. Run the following command to package and upload the buildpack, set up the org and space for tests in the specified CF deployment, and run the CF buildpack tests:
```
$ BUNDLE_GEMFILE=cf.Gemfile buildpack-build
```
If the command fails, you might need to fix or change the tests, fixtures, or other parts of the buildpack.

3. Once the test suite completely passes, use git commands to stage, commit, and push your changes:
```
$ git add .
$ git commit -m "YOUR-MESSAGE[#TRACKER-STORY-ID]"
$ git push
```

4. Monitor the `LANGUAGE-buildpack` pipeline in Concourse. Once the test suite builds, the `specs-lts-develop` job and `specs-edge-develop` job, pass for the buildpack, you can deliver the Tracker story for the new Dependency release. Copy and paste links for the successful test suite builds into the Tracker story.

## Buildpacks
The following list contains information about the buildpacks maintained by the CF buildpacks team:

**Go**:

* **Built from**: A tarred binary, `GO-VERSION.linux-amd64.tar.gz`, provided by [Google on the Go Downloads](https://golang.org/dl/) page.

* **Verified with**: The MD5 of the tarred binary.

* **Example**: [Using the Google Tarred Binary for Go 1.6.2.](https://github.com/cloudfoundry/go-buildpack/blob/v1.7.8/manifest.yml#L33-L34)

**Godep**:

* **Built from**: A source code `.tar.gz` file from the [Godep GitHub releases](https://github.com/tools/godep/releases) page.

* **Verified with**: The SHA256 of the source

**Example**: [Automated enqueuing of binary build for Godep 72.](https://github.com/cloudfoundry/buildpacks-ci/commit/42a361193ea9a4c77a93451228c93bf96f78d50a)
The [buildpacks-ci](https://github.com/cloudfoundry/buildpacks-ci) [dependency-builds](https://github.com/cloudfoundry/buildpacks-ci/blob/master/pipelines/dependency-builds.yml.erb) pipeline automates the process of detecting, uploading, and updating Godep in the manifest.

### Node.js buildpack

**Node**:

* **Verified with**: The SHA256 of the `node-vVERSION.tar.gz` file listed on `https://nodejs.org/dist/vVERSION/SHASUMS256.txt` For example, for Node version 4.4.6, the CF buildpacks team verifies with the SHA256 for `node-v4.4.6.tar.gz` on its [SHASUMS256](https://nodejs.org/dist/v4.4.6/SHASUMS256.txt) page.

* **Example**: [Enqueuing binary builds for Node 4.4.5 and 6.2.0.](https://github.com/cloudfoundry/buildpacks-ci/commit/1747c697dc29db0180253c5d466804784cb3fcce)

### Python buildpack

**Python**:

* **Verified with**: The MD5 of the `Gzipped source tarball`, listed on `https://www.python.org/downloads/release/python-VERSION/`, where `VERSION` has no periods. For example, for Python version `2.7.12`, use the MD5 for the `Gzipped source tarball` on its [downloads](https://www.python.org/downloads/release/python-2712/) page.

* **Example**: [Enqueuing binary build for Python 2.7.12.](https://github.com/cloudfoundry/buildpacks-ci/commit/79e4ce891281ce3ec376b00c967465fc8c7383f8)

### Java buildpack

**OpenJDK**:

* **Built from**: The tarred [OpenJDK files](https://download.run.pivotal.io/openjdk/trusty/x86_64/index.yml) managed by the CF Java Buildpack team.

* **Verified with**: The MD5 of the tarred OpenJDK files.

### Ruby buildpack

**JRuby**:

* **Verified with**: The MD5 of the Source `.tar.gz` file from the [JRuby Downloads](https://www.ruby-lang.org/en/downloads/) page.

* **Example**: [Enqueuing binary build for JRuby 9.1.2.0.](https://github.com/cloudfoundry/buildpacks-ci/commit/1c90967bcb511e7657202a1308e530825e372158)

**Ruby**:

* **Verified with**: The SHA256 of the source from the [Ruby Downloads](https://www.ruby-lang.org/en/downloads/) page.

* **Example**: [Enqueuing binary builds for Ruby 2.2.5 and 2.3.1.](https://github.com/cloudfoundry/buildpacks-ci/commit/8d06eaa6573146b9771de7f96cebf85e07719d79)

**Bundler**:

* **Verified with**: The SHA256 of the `.gem` file from [Rubygems](https://rubygems.org/gems/bundler).

* **Example**: [Enqueuing binary build for Bundler 1.12.5.](https://github.com/cloudfoundry/buildpacks-ci/commit/cd161594a564631dff13e8c101205f922beb5cf8)

### PHP buildpack

**PHP**:

* **Verified with**: The SHA256 of the `.tar.gz` file from the [PHP Downloads](http://php.net/downloads.php) page.

* To enqueue builds for PHP, you need to edit a file in the `public-buildpacks-ci-robots` repository. For PHP5 versions, the CF buildpacks team enqueues builds in the `binary-builds/php-builds.yml` file. For PHP7 versions, the CF buildpacks team enqueues builds in the `binary-builds/php7-builds.yml` file.

* **Example**: [Enqueuing binary builds for PHP 7.2.5 and 7.0.30.](https://github.com/cloudfoundry/public-buildpacks-ci-robots/commit/3353b2d4343f955fdfb34c04358d92c4045aa4e3)

**Nginx**:

* **Verified with**: The `gpg-rsa-key-id` and `gpg-signature` of the version. The `gpg-rsa-key-id` is the same for each version/build, but the `gpg-signature` is different. This information is located on the [Nginx Downloads](https://nginx.org/en/download.html) page.

* **Example**: [Enqueuing binary build for Nginx 1.11.0.](https://github.com/cloudfoundry/buildpacks-ci/commit/b86a09a46eb071957adaecd3a16d1ebba9cd0973)

**HTTPD**:

* **Verified with**: The MD5 of the `.tar.bz2` file from the [HTTPD Downloads](https://httpd.apache.org/download.cgi) page.

* **Example**: [Enqueuing binary build for HTTPD 2.4.20.](https://github.com/cloudfoundry/buildpacks-ci/commit/cee826dc978f013840ef6a8193ec6ee75536778b)

**Composer**:

* **Verified with**: The SHA256 of the `composer.phar` file from the [Composer Downloads](https://getcomposer.org/download/) page.

* For Composer, there is no build process as the `composer.phar` file is the binary. In the manual process, connect to the appropriate S3 bucket using the correct AWS credentials. Create a new directory with the name of the
composer version, for example `1.0.2`, and put the appropriate `composer.phar` file into that directory.
For Composer `v1.0.2`, connect and create the `php/binaries/trusty/composer/1.0.2` directory. Then place the `composer.phar` file into that directory so the binary is available at `php/binaries/trusty/composer/1.0.2/composer.phar`.
The [buildpacks-ci](https://github.com/cloudfoundry/buildpacks-ci) [dependency-builds](https://github.com/cloudfoundry/buildpacks-ci/blob/master/pipelines/dependency-builds.yml.erb) pipeline automates the process of detecting, uploading, and updating Composer in the manifest.

* **Example**: [Automated enqueuing of binary build for Composer 1.1.2.](https://github.com/cloudfoundry/buildpacks-ci/commit/d51abbdb919ca51df86628b37366c55311b05f38)

### Staticfile buildpack

**Nginx**:

* **Verified with**: The `gpg-rsa-key-id` and `gpg-signature` of the version. The `gpg-rsa-key-id` is the same for each version/build, but the `gpg-signature` is different. This information is located on the [Nginx Downloads](https://nginx.org/en/download.html) page.

* **Example**: [Enqueuing binary build for Nginx 1.11.0.](https://github.com/cloudfoundry/buildpacks-ci/commit/b86a09a46eb071957adaecd3a16d1ebba9cd0973)

### Binary buildpack
The Binary buildpack has no dependencies.