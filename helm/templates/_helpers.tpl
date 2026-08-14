{{- define "lightrag.name" -}}
{{- .Chart.Name | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "lightrag.fullname" -}}
{{- if contains .Chart.Name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name .Chart.Name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}

{{- define "lightrag.labels" -}}
app.kubernetes.io/name: {{ include "lightrag.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}

{{- define "lightrag.selectorLabels" -}}
app.kubernetes.io/name: {{ include "lightrag.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}
