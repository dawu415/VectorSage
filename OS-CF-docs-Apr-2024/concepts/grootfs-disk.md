# GrootFS disk usage in Cloud Foundry
This topic tells you about the concepts related to
GrootFS disk space management in Cloud Foundry.

## GrootFS stores
GrootFS is the container root filesystem management component for Garden.
A container root filesystem or *rootfs* is often referred to as an **image**.
A **GrootFS store** is the directory in which rootfs layers and container images are cached.
This directory is configured by GrootFS and mounted on an XFS-formatted volume by the Garden job during BOSH VM creation.
Individual container root filesystems are provided via OverlayFS mounts.
Supplying GrootFS with an already formatted XFS volume for its’ store is not yet supported for BOSH-controlled deployments.

### Garbage collection behavior in GrootFS stores
GrootFS stores are initialized to use the entirety of `/var/vcap/data`. If the `reserved_space_for_other_jobs_in_mb` is not set high enough,
or if there are many images with few shared volumes, the store can use up all available space.
The thresholder component calculates and sets a value so that GrootFS’s garbage collector can attempt to ensure
that a small reserved space is kept free for other jobs.
GrootFS only tries to garbage collect when that threshold is reached.
However, if all the rootfs layers are actively in use by images, then garbage collection cannot occur and that space is used up.

## Volumes
Underlying layers in rootfs images are known as `volumes` in GrootFS.
They are read-only and their changesets are layered together through an **OverlayFS** mount
to create the root filesystems for containers.
When GrootFS writes each filesystem volume to disk, it also stores the number of bytes written
to a file in a `meta` directory.
The size of an individual volume is available in its corresponding metadata file.
GrootFS also stores the SHA of each underlying volume used by an image in the `meta` folder.
For each container, GrootFS mounts the underlying `volumes` using overlay to a point in the `images` directory.
This mount point is the rootfs for the container and is read write.
On disk, the read-write layer for each container can be found at `/var/vcap/data/grootfs/store/unprivileged/images/CONTAINER-ID/diff` (or `/var/vcap/data/grootfs/store/privileged/images/CONTAINER-ID/diff` for privileged containers.)
When GrootFS calls on the built-in XFS quota tooling to get disk usage for a container,
it takes into account data written to those `diff` directories and not the data in the read-only volumes.

### Volume Cleanup Example
When `clean` is called in GrootFS, any layers that are not being used by an existing rootfs are deleted from the store.
The cleanup only takes into account the `volumes` folders in the store.
For example, imagine that there are two rootfs images from different base images, Image A and Image B:
```

- Image A
Layers:

- layer-1

- layer-2

- layer-3

- Image B
Layers:

- layer-1

- layer-4

- layer-5
```
They have a layer in common, layer-1. And after deleting Image B, layer-4 and layer-5 can be collected by clean,
but not layer-1 because Image A still uses that layer.
For more information about how to calculate GrootFS disk usage in your deployment, see [Examining GrootFS Disk Usage](https://docs.cloudfoundry.org/adminguide/examining_grootfs_disk.html).

## Additional information
For more information, see the following sections of `garden-runc-release`:

* [overlay-xfs-setup](https://github.com/cloudfoundry/garden-runc-release/blob/b4a44c5cabb1570eaeb25b158823cfbd97ae530c/jobs/garden/templates/bin/overlay-xfs-setup#L23-L46)

* [grootfs-utils](https://github.com/cloudfoundry/garden-runc-release/blob/b4a44c5cabb1570eaeb25b158823cfbd97ae530c/jobs/garden/templates/bin/grootfs-utils.erb#L31)

* [thresholder](https://github.com/cloudfoundry/garden-runc-release/blob/b4a44c5cabb1570eaeb25b158823cfbd97ae530c/src/thresholder/main.go)