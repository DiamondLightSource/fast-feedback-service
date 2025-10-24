# Fast Feedback Service Helm Chart

This Helm chart deploys the Fast Feedback Service to a Kubernetes cluster.

## Prerequisites

`module load {argus,pollux}`

## Testing the Chart

Before installing, you can test the chart rendering with the test values:

```bash
helm template {name} {folder}
```
For example:
```bash
helm template fast-feedback-service chart/
```
or
```bash
helm template fast-feedback-service chart/ -f chart/values.yaml
```

This will show you the Kubernetes manifests that will be generated without actually installing anything.

## Installing the Chart

To install the chart with the release name `fast-feedback-service`:

```bash
# Using default values (values.yaml)
helm install fast-feedback-service chart/

# Using test values
helm install fast-feedback-service chart/ -f chart/values.test.yaml

# Using custom values
helm install fast-feedback-service chart/ -f chart/{custom_values_file}.yaml
```

## Upgrading the Chart

To upgrade an existing deployment:

```bash
helm upgrade fast-feedback-service chart/ -f chart/values.yaml
```

## Uninstalling the Chart

To remove the deployment:

```bash
helm uninstall fast-feedback-service
```

## Checking the Deployment

Verify the deployment is running:

```bash
kubectl get pods -l app=fast-feedback-service
kubectl logs -l app=fast-feedback-service -f
kubectl describe pod -l app=fast-feedback-service
```