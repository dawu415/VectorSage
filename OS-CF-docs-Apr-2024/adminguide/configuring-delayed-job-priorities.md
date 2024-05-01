# Configuring delayed job priorities with Cloud Controller
You can change the priority of delayed jobs with the Cloud Controller in Cloud Foundry.

## Cloud Controller and job priority
Cloud Foundry creates delayed jobs as it performs certain asynchronous actions, such as `DELETE v3/buildbacks/[GUID]`. These jobs are
processed asynchronously by multiple worker processes.
By default, all jobs have the same priority of `0`. Job priorities of higher numerical values are lower than job priorities of lower numerical values.
Conversely, job priorities of negative numerical values are higher than job priorities of positive numerical values. For example, the Cloud Controller
schedules a job with a priority of `-1` before a job with a priority of `0`, and schedules a job with a priority of `1` before a job with a priority of `0`.
When a job fails, the Cloud Controller might reschedule the job and give it a lower priority, depending on how the job is configured. When the Cloud
Controller reschedules a failed job and gives it a lower priority, it doubles the priority each time the job fails: first from `0` to `1`, then to `2`, then
`4`, then `8`, and so on. Each time a job receives a lower priority, the time until the Cloud Controller schedules its next run increases accordingly. In the
logs for some Cloud Controller components, this scheduled time appears in the `run_at` column of the `delayed_jobs` table.
When a job with a priority of negative numerical value fails multiple times, the Cloud Controller lowers its priority to `0`, then doubles its priority
afterward.

## Delayed jobs
In the logs for some Cloud Controller components, you can view a jobs that have been delayed in the `delayed_jobs` table.
The following list contains the `display_name` of each job in the `delayed_jobs` table that you can configure with a different default priority:

* `service_binding.delete`

* `organization.delete`

* `space.delete`

* `service_instance.delete`

* `service_key.delete`

* `service_key.delete`

* `service_key.delete`

* `app_model.delete`

* `buildpack.delete`

* `domain.delete`

* `droplet_model.delete`

* `droplet_model.delete`

* `quota_definition.delete`

* `packages_model.delete`

* `role.delete`

* `route.delete`

* `security_group.delete`

* `service_broker.delete`

* `space_quota_definition.delete`

* `user.delete`

* `space.apply_manifest`

* `admin.clear_buildpack_cache`

* `service_instance.create`

* `service_bindings.create`

* `buildpack.upload`

* `space.delete_unmapped_routes`

* `service_keys.delete`

* `service_instance.update`

* `service_route_bindings.create`

* `service_route_bindings.delete`

* `service_keys.create`

* `droplet.upload`

* `service_bindings.delete`

* `service_broker.catalog.synchronize`

* `service_broker.update`

## Overriding the default priority of a job
You can override the default priority for the jobs listed in [Delayed Jobs](https://docs.cloudfoundry.org/adminguide/configuring-delayed-job-priorities.html#delayed-jobs) above by creating an operations file in YAML format that contains a
list of job names and their new priorities.
The following example shows an operations file that gives the `space.apply_manifest` job the highest priority and configures the `cloud_controller_ng` job to
run it before any other job:
```

- type: replace
path: /instance_groups/name=api/jobs/name=cloud_controller_ng/properties/cc/jobs?/priorities?
value:
space.apply_manifest: -10
```
For more information about operations files, see the [BOSH documentation](https://bosh.io/docs/cli-ops-files/).