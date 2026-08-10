import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { PauseIcon, PlayIcon, RefreshCwIcon, Trash2Icon } from 'lucide-react'

import Button from '@/components/ui/Button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/Card'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue
} from '@/components/ui/Select'
import { getLogFiles, streamLogFile, type LogFileInfo, type LogStreamChunk } from '@/api/lightrag'
import { errorMessage } from '@/lib/utils'

const MAX_RENDERED_LINES = 2000

function appendLines(previous: string[], incoming: string[]) {
  const next = [...previous, ...incoming]
  if (next.length <= MAX_RENDERED_LINES) {
    return next
  }
  return next.slice(next.length - MAX_RENDERED_LINES)
}

export default function LogsViewer() {
  const { t } = useTranslation()
  const [files, setFiles] = useState<LogFileInfo[]>([])
  const [selectedFileId, setSelectedFileId] = useState('')
  const [lines, setLines] = useState<string[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [isStreaming, setIsStreaming] = useState(false)
  const [isPaused, setIsPaused] = useState(false)
  const [autoScroll, setAutoScroll] = useState(true)
  const [statusText, setStatusText] = useState('')
  const [errorText, setErrorText] = useState('')
  const [activePath, setActivePath] = useState('')
  const [lastUpdateAt, setLastUpdateAt] = useState<number | null>(null)
  const abortRef = useRef<AbortController | null>(null)
  const bottomRef = useRef<HTMLDivElement | null>(null)

  const selectedFile = useMemo(
    () => files.find((file) => file.id === selectedFileId) ?? null,
    [files, selectedFileId]
  )

  const stopStreaming = useCallback(() => {
    abortRef.current?.abort()
    abortRef.current = null
    setIsStreaming(false)
  }, [])

  const loadFiles = useCallback(async () => {
    try {
      setIsLoading(true)
      setErrorText('')
      const payload = await getLogFiles()
      setFiles(payload.files)
      setSelectedFileId((current) => {
        if (current && payload.files.some((file) => file.id === current)) {
          return current
        }
        return payload.default_file_id || payload.files[0]?.id || ''
      })
    } catch (error) {
      setErrorText(errorMessage(error))
    } finally {
      setIsLoading(false)
    }
  }, [])

  const handleStreamChunk = useCallback((chunk: LogStreamChunk) => {
    setLastUpdateAt(Date.now())

    if (chunk.path) {
      setActivePath(chunk.path)
    }

    if (chunk.type === 'snapshot' || chunk.type === 'reset') {
      setLines(chunk.lines || [])
      setStatusText(
        chunk.type === 'reset'
          ? t('logsPanel.status.reset', 'Log file rotated or truncated, view was reloaded.')
          : t('logsPanel.status.connected', 'Connected to live log stream.')
      )
      return
    }

    if (chunk.type === 'append') {
      setLines((current) => appendLines(current, chunk.lines || []))
      setStatusText(t('logsPanel.status.streaming', 'Receiving new log lines in realtime.'))
      return
    }

    if (chunk.type === 'error') {
      setErrorText(chunk.message || t('logsPanel.status.error', 'Log stream failed.'))
      setStatusText(t('logsPanel.status.disconnected', 'Log stream disconnected.'))
      setIsStreaming(false)
    }
  }, [t])

  useEffect(() => {
    loadFiles()
  }, [loadFiles])

  useEffect(() => {
    if (!selectedFileId || isPaused) {
      stopStreaming()
      return
    }

    setLines([])
    setErrorText('')
    setStatusText(t('logsPanel.status.connecting', 'Connecting to log stream...'))

    const controller = new AbortController()
    abortRef.current = controller
    setIsStreaming(true)

    void streamLogFile(
      selectedFileId,
      handleStreamChunk,
      (message) => {
        if (!controller.signal.aborted) {
          setErrorText(message)
          setStatusText(t('logsPanel.status.disconnected', 'Log stream disconnected.'))
          setIsStreaming(false)
        }
      },
      {
        tailLines: 200,
        signal: controller.signal
      }
    ).finally(() => {
      if (!controller.signal.aborted) {
        setIsStreaming(false)
      }
    })

    return () => {
      controller.abort()
      if (abortRef.current === controller) {
        abortRef.current = null
      }
    }
  }, [handleStreamChunk, isPaused, selectedFileId, stopStreaming, t])

  useEffect(() => {
    if (!autoScroll) {
      return
    }
    bottomRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [autoScroll, lines])

  const connectionClassName = isStreaming
    ? 'bg-emerald-500/15 text-emerald-700 dark:text-emerald-300'
    : 'bg-zinc-500/15 text-zinc-700 dark:text-zinc-300'

  return (
    <div className="h-full overflow-auto p-4 md:p-6">
      <Card className="flex min-h-full flex-col">
        <CardHeader className="gap-4 border-b pb-4">
          <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
            <div className="space-y-1">
              <CardTitle>{t('logsPanel.title', 'Live Logs')}</CardTitle>
              <CardDescription>
                {t(
                  'logsPanel.description',
                  'Stream LightRAG backend logs directly in the WebUI with automatic updates.'
                )}
              </CardDescription>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <span className={`rounded-full px-2.5 py-1 text-xs font-medium ${connectionClassName}`}>
                {isStreaming
                  ? t('logsPanel.badges.live', 'Live')
                  : t('logsPanel.badges.idle', 'Idle')}
              </span>
              {lastUpdateAt && (
                <span className="text-muted-foreground text-xs">
                  {t('logsPanel.lastUpdate', 'Updated')} {new Date(lastUpdateAt).toLocaleTimeString()}
                </span>
              )}
            </div>
          </div>

          <div className="flex flex-col gap-3 xl:flex-row xl:items-center">
            <div className="min-w-0 flex-1">
              <Select
                value={selectedFileId}
                onValueChange={(value) => {
                  setSelectedFileId(value)
                  setErrorText('')
                }}
                disabled={isLoading || files.length === 0}
              >
                <SelectTrigger className="w-full">
                  <SelectValue placeholder={t('logsPanel.selectPlaceholder', 'Select a log file')} />
                </SelectTrigger>
                <SelectContent>
                  {files.map((file) => (
                    <SelectItem key={file.id} value={file.id}>
                      {file.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="flex flex-wrap gap-2">
              <Button
                variant="outline"
                size="sm"
                tooltip={t('logsPanel.refreshTooltip', 'Refresh available log files')}
                onClick={() => void loadFiles()}
              >
                <RefreshCwIcon />
                {t('logsPanel.refresh', 'Refresh')}
              </Button>
              <Button
                variant="outline"
                size="sm"
                tooltip={isPaused ? t('logsPanel.resumeTooltip', 'Resume log streaming') : t('logsPanel.pauseTooltip', 'Pause log streaming')}
                disabled={!selectedFileId}
                onClick={() => setIsPaused((value) => !value)}
              >
                {isPaused ? <PlayIcon /> : <PauseIcon />}
                {isPaused ? t('logsPanel.resume', 'Resume') : t('logsPanel.pause', 'Pause')}
              </Button>
              <Button
                variant="outline"
                size="sm"
                tooltip={t('logsPanel.clearTooltip', 'Clear only the current UI buffer')}
                onClick={() => setLines([])}
              >
                <Trash2Icon />
                {t('logsPanel.clear', 'Clear View')}
              </Button>
              <Button
                variant={autoScroll ? 'default' : 'outline'}
                size="sm"
                tooltip={t('logsPanel.autoScrollTooltip', 'Automatically follow new log lines')}
                onClick={() => setAutoScroll((value) => !value)}
              >
                {t('logsPanel.autoScroll', 'Auto-scroll')}
              </Button>
            </div>
          </div>

          <div className="grid gap-2 text-sm md:grid-cols-[minmax(0,1fr)_auto_auto] md:items-center">
            <div className="min-w-0">
              <div className="text-muted-foreground text-xs uppercase tracking-wide">
                {t('logsPanel.activeFile', 'Active File')}
              </div>
              <div className="truncate font-mono text-xs">
                {activePath || selectedFile?.path || t('logsPanel.noFile', 'No log file selected')}
              </div>
            </div>
            <div className="text-muted-foreground text-xs">
              {t('logsPanel.lineCount', 'Lines in view')}: {lines.length}
            </div>
            <div className="text-muted-foreground text-xs">
              {statusText}
            </div>
          </div>

          {errorText && (
            <div className="rounded-md border border-red-500/30 bg-red-500/10 px-3 py-2 text-sm text-red-700 dark:text-red-300">
              {errorText}
            </div>
          )}
        </CardHeader>

        <CardContent className="flex min-h-0 flex-1 flex-col p-0">
          <div className="bg-zinc-950 text-zinc-100 min-h-0 flex-1 overflow-auto rounded-b-xl">
            <pre className="min-h-full p-4 font-mono text-xs leading-5 whitespace-pre-wrap break-words">
              {lines.length > 0
                ? lines.join('\n')
                : isLoading
                  ? t('logsPanel.loading', 'Loading log files...')
                  : t('logsPanel.empty', 'No log output yet.')}
              <div ref={bottomRef} />
            </pre>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
