# Packaging dependencies for offline buildpacks
Learn about the dependency storage options that are available to you when creating offline buildpacks.

## About offline buildpacks
Online, or uncached, buildpacks require an internet connection to download dependencies.
For example, language interpreters, and compilers. Alternatively, you can create offline, or cached, buildpacks that are packaged with their dependencies. These offline buildpacks do not connect to the Internet when they are used to deploy Cloud Foundry apps.
Offline buildpacks might contain proprietary dependencies that require distribution licensing or export control measures.
You can find instructions for building offline packages in the `README.md` file for each buildpack repository. For example, see the [Java buildpack](https://github.com/cloudfoundry/java-buildpack#offline-package).

## Packaging dependencies in the buildpack
A simple way to package dependencies for a custom buildpack is to keep the dependencies in your buildpack source.
However, this is not recommended. Keeping the dependencies in your source consumes unnecessary space.
To avoid keeping the dependencies in source control, load the dependencies into your buildpack, and
provide a script for the operator to create a zipfile of the buildpack.
For example, you might complete the following process:
```
$ # Clones your buildpack
$ git clone http://YOUR-GITHUB-REPOSITORY.example.com/repository
$ cd SomeBuildPackName
$ # Creates a zipfile using your script
$ ./SomeScriptName

----> downloading-dependencies.... done

----> creating zipfile: ZippedBuildPackName.zip
$ # Adds the buildpack zipfile to the Cloud Foundry instance
$ cf create-buildpack SomeBuildPackName ZippedBuildPackName.zip 1
```

### Pros

* Least complicated process for operators.

* Least complicated maintenance process for buildpack developers.

### Cons

* Cloud Foundry admin buildpack uploads are limited to 1 GB, so the dependencies might not fit.

* Security and functional patches to dependencies require updating the buildpack.

## Packaging selected dependencies in the buildpack
This is a variant of the [package dependencies in the buildpack](https://docs.cloudfoundry.org/buildpacks/depend-pkg-offline.html#package-directly) method. In this variation, the administrator edits a configuration file. For example, the `dependencies.yml` file to include a limited subset of the buildpack dependencies,
packages, and uploads to the buildpack.
This approach is not recommended. See the Cons section for more information.
The administrator completes the following steps:
```
$ # Clones your buildpack
$ git clone http://YOUR-GITHUB-REPOSITORY.example.com/repository
$ cd SomeBuildPackName
$ # Selects dependencies
$ vi dependencies.yml # Or copy in a preferred config
$ # Builds a package using your script
$ ./package

----> downloading-dependencies.... done

----> creating zipfile: cobol_buildpack.zip
$ # Adds the buildpack to the Cloud Foundry instance
$ cf create-buildpack cobol-buildpack cobol_buildpack.zip 1
$ # Pushes an app using your buildpack
$ cd ~/my_app
$ cf push my-cobol-webapp -b cobol-buildpack
```

### Pros

* It’s possible to avoid the Cloud Foundry admin buildpack upload size limit in the following ways:

+ The administrator chooses a limited subset of dependencies.

+ The administrator maintains different packages for different dependency sets.

### Cons

* More complex for buildpack maintainers.

* Security updates to dependencies require updating the buildpack.

* Proliferation of buildpacks require maintenance:

+ For each configuration, there is an update required for each security patch.

+ Culling orphan configurations can be difficult or impossible.

+ Administrators need to track configurations and merge them with updates to the buildpack.

+ Might result in a different configuration for each app.

## Relying on a local mirror
In this method, the administrator provides a compatible file store of dependencies. When running the buildpack, the administrator specifies the location of the file store.
The administrator completes the following process:
```
$ # Clones your buildpack
$ git clone http://YOUR-GITHUB-REPOSITORY.example.com/repository
$ cd SomeBuildPackName
$ # Builds a package using your script
$ ./package https:///dependency/repository

----> creating zipfile: cobol_buildpack.zip
$ # Adds the buildpack to the Cloud Foundry instance
$ cf create-buildpack cobol-buildpack cobol_buildpack.zip 1
$ # Pushes an app using your buildpack
$ cd ~/my_app
$ cf push my-cobol-webapp -b cobol-buildpack

----> deploying app

----> downloading dependencies:
https://OUR-INTERNAL-SITE.example.com/dependency/repository/dep1.tgz.... done
https://OUR-INTERNAL-SITE.example.com/dependency/repository/dep2.tgz.... WARNING: dependency not found!
```

### Pros

* Avoids the Cloud Foundry admin buildpack upload size limit.

* Leaves the administrator completely in control of providing dependencies.

* Security and functional patches for dependencies can be maintained separately on the mirror given the following conditions:

+ The buildpack is designed to use newer semantically versioned dependencies.

+ Buildpack behavior does not change with the newer functional changes.

### Cons

* The administrator needs to set up and maintain a mirror.

* The additional config option presents a maintenance burden.