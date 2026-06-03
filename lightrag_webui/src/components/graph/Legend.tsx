import React from 'react'
import { useTranslation } from 'react-i18next'
import { useGraphStore } from '@/stores/graph'
import { Card } from '@/components/ui/Card'
import { ScrollArea } from '@/components/ui/ScrollArea'

interface LegendProps {
  className?: string
}

const Legend: React.FC<LegendProps> = ({ className }) => {
  const { t } = useTranslation()
  const typeColorMap = useGraphStore.use.typeColorMap()
  const activeLegendTypes = useGraphStore.use.activeLegendTypes()
  const toggleLegendType = useGraphStore.use.toggleLegendType()
  const clearLegendTypeFilter = useGraphStore.use.clearLegendTypeFilter()

  if (!typeColorMap || typeColorMap.size === 0) {
    return null
  }

  return (
    <Card className={`flex max-h-80 max-w-xs flex-col p-2 ${className}`}>
      <h3 className="mb-2 text-sm font-medium">{t('graphPanel.legend')}</h3>
      {activeLegendTypes.length > 0 && (
        <button
          className="mb-2 text-left text-xs text-muted-foreground hover:text-foreground"
          onClick={clearLegendTypeFilter}
          type="button"
        >
          Clear filter
        </button>
      )}
      <ScrollArea className="h-80 pr-2">
        <div className="flex flex-col gap-1">
          {Array.from(typeColorMap.entries()).map(([type, color]) => (
            <button
              key={type}
              className={`flex w-full items-center gap-2 rounded px-1 py-1 text-left transition-colors ${
                activeLegendTypes.length === 0 || activeLegendTypes.includes(type)
                  ? 'bg-accent/40 text-foreground'
                  : 'opacity-45 hover:opacity-80'
              }`}
              onClick={() => toggleLegendType(type)}
              type="button"
            >
              <div
                className="w-4 h-4 rounded-full"
                style={{ backgroundColor: color }}
              />
              <span className="text-xs truncate" title={type}>
                {t(`graphPanel.nodeTypes.${type.toLowerCase().replace(/\s+/g, '')}`, type)}
              </span>
            </button>
          ))}
        </div>
      </ScrollArea>
    </Card>
  )
}

export default Legend
