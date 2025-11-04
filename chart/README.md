# Fast Feedback Service Helm Chart

This Helm chart deploys the Fast Feedback Service to a Kubernetes cluster.

## Prerequisites

`module load {argus,pollux,k8s-i24}`

## Understanding Helm Commands

- `helm template` - Preview manifests locally (doesn't deploy anything)
- `helm install` - **Deploys to the cluster** (creates real running pods)
- `helm upgrade` - Update an existing deployment
- `helm uninstall` - **Remove from cluster** (deletes all resources)

## Testing the Chart Locally

Before installing, you can test the chart rendering with the test values:

```bash
# Preview what will be deployed (dry-run)
helm template ffs chart/ -f chart/values.test.yaml

# Or with debug output
helm install ffs chart/ -f chart/values.test.yaml --dry-run --debug
```

This will show you the Kubernetes manifests that will be generated **without actually deploying anything**.

## Installing the Chart

To install the chart and **deploy to the cluster**:

```bash
# Using upgrade --install (recommended - installs or upgrades as needed)
helm -n i24-beamline upgrade --install ffs-test chart/

# With custom values file
helm -n i24-beamline upgrade --install ffs-test chart/ -f chart/values.test.yaml
```

**Note:** The `-n i24-beamline` flag specifies the Kubernetes namespace. Always include this to deploy to the correct namespace.

## Checking the Deployment

Verify the deployment is running:

```bash
# List all Helm releases in the namespace
helm -n i24-beamline list

# Check pod status
kubectl -n i24-beamline get pods

# Watch pod status in real-time
kubectl -n i24-beamline get pods -w

# View logs from deployment
kubectl -n i24-beamline logs deployment/ffs

# View logs from previous pod (if pod crashed/restarted)
kubectl -n i24-beamline logs deployment/ffs --previous

# View logs from a specific pod (get pod name from 'kubectl get pods')
kubectl -n i24-beamline logs <pod-name>

# Get detailed pod information
kubectl -n i24-beamline describe pod <pod-name>
```

## Upgrading the Chart

To upgrade an existing deployment:

```bash
# Upgrade with values (this is the same command as install)
helm -n i24-beamline upgrade --install ffs-test chart/

# With custom values file
helm -n i24-beamline upgrade --install ffs-test chart/ -f chart/values.yaml

# Force recreation of pods
helm -n i24-beamline upgrade --install ffs-test chart/ --force
```

## Uninstalling the Chart

To **remove the deployment from the cluster**:

```bash
helm -n i24-beamline uninstall ffs-test

# Verify removal
kubectl -n i24-beamline get pods
helm -n i24-beamline list
```

## Troubleshooting

### Pod won't start

```bash
# First, get the pod name
kubectl -n i24-beamline get pods

# Check pod status and events
kubectl -n i24-beamline describe pod <pod-name>

# View logs from current pod
kubectl -n i24-beamline logs <pod-name>

# View logs from previous pod (if it crashed)
kubectl -n i24-beamline logs deployment/ffs --previous
