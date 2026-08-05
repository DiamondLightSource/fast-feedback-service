{{/*
Pod-level scheduling and storage shared by the Deployment and the
batch Jobs, so that they cannot drift apart.
*/}}

{{- define "ffs.nodeSelector" -}}
nodeSelector:
  {{- range $key, $value := .Values.nodeSelector }}
  {{ $key }}: {{ $value | quote }}
  {{- end }}
{{- end }}

{{- define "ffs.tolerations" -}}
tolerations:
  - key: nodetype
    value: gpu
    effect: NoSchedule
  - key: location
    value: cs05r
    effect: NoSchedule
{{- end }}

{{- define "ffs.volumes" -}}
volumes:
  - name: dls
    hostPath:
      path: /dls/{{ .Values.beamline }}
      type: Directory
  - name: dlssw
    hostPath:
      path: /dls_sw/apps
      type: Directory
{{- end }}

{{/*
Mounts for the beamline filesystem. The batch Jobs write their
results under /dls, so they take the mount read-write; the service
only reads.
*/}}
{{- define "ffs.volumeMounts" -}}
volumeMounts:
  - mountPath: /dls/{{ .Values.beamline }}
    name: dls
    mountPropagation: HostToContainer
    readOnly: {{ .readOnlyDls }}
  - mountPath: /dls_sw/apps
    name: dlssw
    mountPropagation: HostToContainer
    readOnly: true
{{- end }}

{{- define "ffs.resources" -}}
resources:
  limits:
    cpu: {{ .Values.resources.cpu }}
    memory: {{ .Values.resources.memory }}
    nvidia.com/gpu: {{ .Values.resources.gpu }}
{{- end }}
