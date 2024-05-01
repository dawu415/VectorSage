# Garden component
You can use Garden, the component that Cloud Foundry uses to create and manage isolated environments
called containers. Each instance of an app deployed to Cloud Foundry runs within a container.
For more information about how containers work, see [Container Mechanics](https://docs.cloudfoundry.org/concepts/container-security.html#mechanics) in *Container Security*.

## Plug-in back ends
Garden has plug-in back ends for different platforms and runtimes. It
specifies a set of interfaces that each platform specific back end must implement.
These interfaces contain methods to perform the following actions:

* Create and delete containers.

* Apply resource limits to containers.

* Open and attach network ports to containers.

* Copy files into and out of containers.

* Run processes within containers.

* Stream `STDOUT` and `STDERR` data out of containers.

* Annotate containers with arbitrary metadata.

* Snapshot containers for redeploys without downtime.
For more information, see the [Garden](https://github.com/cloudfoundry/garden) repository on GitHub.

## Garden-runC back end
Cloud Foundry currently uses the Garden-runC back end, a Linux-specific implementation of the Garden interface using the [Open Container Interface](https://github.com/opencontainers/runtime-spec) (OCI) standard. Previous versions of Cloud Foundry used the Garden-Linux back end. For more information, see the [Garden-Linux](https://github.com/cloudfoundry-attic/garden-linux) repository on GitHub.
Garden-runC has the following features:

* Uses the same OCI low-level container execution code as Docker and Kubernetes, so container images run identically across all three platforms

* [AppArmor](https://wiki.ubuntu.com/AppArmor) is configured and enforced by default for all unprivileged containers

* Seccomp allowlisting restricts the set of system calls a container can access, reducing the risk of container breakout

* Allows plug-in networking and rootfs management
For more information, see the [Garden-runC](https://github.com/cloudfoundry/garden-runc-release) repository on GitHub.

## Garden RootFS plug-in
Garden manages container file systems through a plug-in interface. Cloud Foundry uses the Garden RootFS (GrootFS) plug-in for this task.
GrootFS is a Linux-specific implementation of the Garden volume plug-in interface.
GrootFS performs the following actions:

* Creates container file systems based on buildpacks and droplets.

* Creates container file systems based on remote docker images.

* Authenticates with remote registries when using remote images.

* Properly maps UID/GID for all files inside an image.

* Runs garbage collection to remove unused volumes.

* Applies per container disk quotas.

* Provides per container disk usage stats.
For more information, see [GrootFS Disk Usage](https://docs.cloudfoundry.org/concepts/grootfs-disk.html) and the [GrootFS repository](https://github.com/cloudfoundry/grootfs) on GitHub.