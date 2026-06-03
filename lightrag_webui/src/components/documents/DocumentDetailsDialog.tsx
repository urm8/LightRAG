import { useTranslation } from 'react-i18next'
import Button from '@/components/ui/Button'
import Input from '@/components/ui/Input'
import Textarea from '@/components/ui/Textarea'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle
} from '@/components/ui/Dialog'
import { DocStatusResponse } from '@/api/lightrag'
import { RotateCcwIcon, TrashIcon } from 'lucide-react'

type DocumentDetailsDialogProps = {
  document: DocStatusResponse | null
  open: boolean
  onOpenChange: (open: boolean) => void
  onDelete: (document: DocStatusResponse) => Promise<void>
  onReindex: (document: DocStatusResponse) => Promise<void>
  actionInProgress: 'delete' | 'reindex' | null
}

const getStatusClassName = (status: DocStatusResponse['status']): string => {
  switch (status) {
    case 'processed':
      return 'text-green-600'
    case 'preprocessed':
      return 'text-purple-600'
    case 'processing':
      return 'text-blue-600'
    case 'pending':
      return 'text-yellow-600'
    case 'failed':
      return 'text-red-600'
    default:
      return 'text-foreground'
  }
}

const getStatusLabel = (
  status: DocStatusResponse['status'],
  t: ReturnType<typeof useTranslation>['t']
): string => {
  switch (status) {
    case 'processed':
      return t('documentPanel.documentManager.status.completed')
    case 'preprocessed':
      return t('documentPanel.documentManager.status.preprocessed')
    case 'processing':
      return t('documentPanel.documentManager.status.processing')
    case 'pending':
      return t('documentPanel.documentManager.status.pending')
    case 'failed':
      return t('documentPanel.documentManager.status.failed')
    default:
      return status
  }
}

export default function DocumentDetailsDialog({
  document,
  open,
  onOpenChange,
  onDelete,
  onReindex,
  actionInProgress
}: DocumentDetailsDialogProps) {
  const { t } = useTranslation()

  if (!document) {
    return null
  }

  const isDeleting = actionInProgress === 'delete'
  const isReindexing = actionInProgress === 'reindex'
  const isBusy = actionInProgress !== null
  const hasFilePath = Boolean(document.file_path?.trim())
  const fileName = document.file_path.split('/').filter(Boolean).pop() || document.file_path || document.id

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-2xl" onCloseAutoFocus={(event) => event.preventDefault()}>
        <DialogHeader>
          <DialogTitle>{fileName}</DialogTitle>
          <DialogDescription>{document.file_path || document.id}</DialogDescription>
        </DialogHeader>

        <div className="space-y-4">
          <div className="space-y-2">
            <div className="text-sm font-medium">{t('documentPanel.documentManager.fileNameLabel')}</div>
            <Input value={fileName} readOnly />
          </div>

          <div className="grid gap-4 sm:grid-cols-2">
            <div className="space-y-2">
              <div className="text-sm font-medium">{t('documentPanel.documentManager.columns.status')}</div>
              <Input
                className={getStatusClassName(document.status)}
                value={getStatusLabel(document.status, t)}
                readOnly
              />
            </div>
            <div className="space-y-2">
              <div className="text-sm font-medium">{t('documentPanel.documentManager.columns.length')}</div>
              <Input value={String(document.content_length ?? '-')} readOnly />
            </div>
          </div>

          <div className="space-y-2">
            <div className="text-sm font-medium">{t('documentPanel.documentManager.columns.summary')}</div>
            <Textarea
              value={document.content_summary || '-'}
              readOnly
              className="min-h-32"
            />
          </div>
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)} disabled={isBusy}>
            {t('common.cancel')}
          </Button>
          <Button
            variant="destructive"
            onClick={() => onDelete(document)}
            disabled={isBusy}
          >
            <TrashIcon />
            {isDeleting ? t('documentPanel.deleteDocuments.deleting') : t('documentPanel.deleteDocuments.button')}
          </Button>
          <Button
            variant="secondary"
            onClick={() => onReindex(document)}
            disabled={isBusy || !hasFilePath}
          >
            <RotateCcwIcon />
            {isReindexing
              ? t('documentPanel.documentManager.reindexing', { defaultValue: 'Reindexing...' })
              : t('documentPanel.documentManager.reindexButton', { defaultValue: 'Reindex' })}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}