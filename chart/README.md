# Fast Feedback Service Helm Chart

This Helm chart deploys the Fast Feedback Service to a Kubernetes cluster.

## Prerequisites

`module load {argus,pollux,k8s-i24}`

## Namespace Configuration

When deploying to a beamline cluster, you need to work in the correct namespace. You have two options:

**Option 1: Set the namespace context once** (recommended)
```bash
kubectl config set-context --current --namespace=i24-beamline
```

**Option 2: Add `-n i24-beamline` to every kubectl command**
```bash
kubectl get pods -n i24-beamline
kubectl logs -n i24-beamline ffs-5486c998bc-cnskz
```

The examples below assume you've set the namespace context. If you haven't, add `-n i24-beamline` to each kubectl command.

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
# Using default values (values.yaml)
helm install ffs chart/

# Using test values
helm install ffs chart/ -f chart/values.test.yaml

# Targeting a specific node (e.g., for testing on a GPU node)
helm install ffs chart/ \
  -f chart/values.test.yaml \
  --set nodeSelector."kubernetes\.io/hostname"="cs05r-i24-k8s-serv-02.diamond.ac.uk"

# Using custom values
helm install ffs chart/ -f chart/{custom_values_file}.yaml
```

## Checking the Deployment

First, find your pod name, then use it for all subsequent commands:

```bash
# List all Helm releases
helm list

# Find your pod (look for the one starting with your release name)
kubectl get pods

# Example output:
# NAME                          READY   STATUS    RESTARTS   AGE
# ffs-5486c998bc-cnskz          1/1     Running   0          5m
```

Once you have the pod name (e.g., `ffs-5486c998bc-cnskz`), use these commands:

```bash
# Check pod status
kubectl get pod ffs-5486c998bc-cnskz

# Watch pod status in real-time
kubectl get pod ffs-5486c998bc-cnskz -w

# View logs
kubectl logs ffs-5486c998bc-cnskz

# Follow logs in real-time
kubectl logs -f ffs-5486c998bc-cnskz

# View previous logs (if pod restarted)
kubectl logs ffs-5486c998bc-cnskz --previous

# Get detailed pod information
kubectl describe pod ffs-5486c998bc-cnskz

# Execute commands inside the pod
kubectl exec ffs-5486c998bc-cnskz -- env | grep -E 'SPOTFINDER|ZOCALO'

# Check recent cluster events
kubectl get events --sort-by='.lastTimestamp'
```

## Upgrading the Chart

To upgrade an existing deployment:

```bash
# Upgrade with new values
helm upgrade ffs chart/ -f chart/values.yaml

# Force recreation of pods
helm upgrade ffs chart/ -f chart/values.yaml --force

# After upgrade, find the new pod name
kubectl get pods
```

**Note:** After an upgrade, the pod name will change (new hash suffix).

## Uninstalling the Chart

To **remove the deployment from the cluster**:

```bash
helm uninstall ffs

# Verify removal
kubectl get pods
helm list
```