# MLOps Fraud Detection Pattern

## Installation Instructions

1. Create an OpenShift 4 cluster.
2. Log into the cluster using `oc login` or by exporting your `KUBECONFIG`.
3. Clone this repository and `cd` into the root folder (where this README is located).
4. Install the pattern by running: `./pattern.sh make install`
5. Wait for all components to deploy via Argo CD. You’ll know everything is ready when the `Hub ArgoCD` instance (accessible via the 9-dots menu) shows all applications as **healthy** and **synced**.

## Notable Links

1. In the 9-dots menu, you’ll also see **Inferencing App**, a small Gradio-based web app for testing the trained model. It includes two test cases: the first is *not fraudulent*, and the second *is fraudulent*. The app source code is located at [src/inferencing-app/app.py](./src/inferencing-app/app.py).

2. You’ll also find a link to **Red Hat OpenShift AI** in the 9-dots menu. Most of the resources for this pattern are deployed to the OpenDataHub dashboard, within the `fraud-detection` project/namespace.

   Of particular note:
   - A pipeline named `fraud-detection` will already have completed its first run by the time installation finishes.
   - Navigate to `Models → Model deployments` in the left-hand menu to find a `fraud-detection` model deployment. This deployment is actively serving the model produced by that initial pipeline run.
