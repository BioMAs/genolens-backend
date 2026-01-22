#!/bin/bash
BACKUP_DIR="/home/dev/backups/genolens"
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p $BACKUP_DIR

# Backup PostgreSQL
echo "💾 Backup de la base de données..."
cd /home/dev/genolens_v2/backend && docker compose -f docker-compose.prod.yml exec -T postgres pg_dump -U genolens genolens_production | gzip > $BACKUP_DIR/db_$DATE.sql.gz

# Backup volumes (data files)
echo "💾 Backup des volumes..."
docker run --rm -v backend_genolens_storage:/data -v $BACKUP_DIR:/backup alpine tar czf /backup/data_$DATE.tar.gz -C /data .

# Cleanup old backups (garder 7 jours)
find $BACKUP_DIR -name "*.gz" -mtime +7 -delete

echo "✅ Backup terminé: $BACKUP_DIR"
ls -lh $BACKUP_DIR/*$DATE*
