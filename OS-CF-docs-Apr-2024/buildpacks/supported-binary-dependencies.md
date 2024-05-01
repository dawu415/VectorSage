# Supported binary dependencies in Cloud Foundry buildpacks
Each buildpack supports only the stable patches for each dependency listed in
the buildpack’s `manifest.yml`file and also in the GitHub releases page.
For example, see the [php-buildpack releases page](https://github.com/cloudfoundry/php-buildpack/releases).
If you try to use an unsupported binary, staging your app fails with the following error message:
```
Could not get translated url, exited with: DEPENDENCY_MISSING_IN_MANIFEST:
...
!
! exit
!
Staging failed: Buildpack compilation step failed
```