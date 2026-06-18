import type { DocStatus, DocumentsRequest } from '@/api/lightrag'

export type StatusBucket = 'pending' | 'completed' | 'parse' | 'analyze' | 'process' | 'failed'
export type StatusFilter = StatusBucket | 'all'

// Each filter bucket maps to exactly one DocStatus. `preprocessed` intentionally
// has no dedicated bucket — it only surfaces under the "all" tab.
const BUCKET_TO_STATUS: Record<StatusBucket, DocStatus> = {
  pending: 'pending',
  completed: 'processed',
  parse: 'parsing',
  analyze: 'analyzing',
  process: 'processing',
  failed: 'failed'
}

const STATUS_TO_BUCKET: Partial<Record<DocStatus, StatusBucket>> = {
  pending: 'pending',
  processed: 'completed',
  parsing: 'parse',
  analyzing: 'analyze',
  processing: 'process',
  failed: 'failed'
}

export const getStatusBucket = (status: DocStatus): StatusBucket | null =>
  STATUS_TO_BUCKET[status] ?? null

export const getStatusRequestFilters = (
  statusFilter: StatusFilter
): Pick<DocumentsRequest, 'status_filter' | 'status_filters'> => {
  if (statusFilter === 'all') {
    return {
      status_filter: null,
      status_filters: null
    }
  }

  return {
    status_filter: BUCKET_TO_STATUS[statusFilter],
    status_filters: null
  }
}

export const matchesStatusFilter = (status: DocStatus, statusFilter: StatusFilter): boolean =>
  statusFilter === 'all' || BUCKET_TO_STATUS[statusFilter] === status
