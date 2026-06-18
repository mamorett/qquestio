package rag

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// CorpusCache is the on-disk cache for a Qdrant collection's full corpus.
// It is written by the first full-corpus search and read by every subsequent
// search of the same collection, so we never re-scroll the entire index
// unless the cache is invalidated, the collection changed, or the user
// explicitly requests a refresh.
//
// The file format is intentionally simple: a small binary header followed by
// all payloads as one JSON blob and all vectors as raw float32 bytes. This
// trades a tiny bit of space for much faster load/save than 1M tiny JSON
// objects.
type CorpusCache struct {
	Collection string    `json:"collection"`
	Dimension  int       `json:"dimension"`
	PointCount int       `json:"point_count"`
	CachedAt   time.Time `json:"cached_at"`
	filePath   string
}

const (
	cacheMagic   uint32 = 0x51434341 // "QQCA" in little-endian
	cacheVersion uint8  = 1
)

// CacheDir returns the directory where corpus caches are stored.
// Honors the QQUESTIO_CACHE_DIR env var; defaults to $HOME/.cache/qquestio.
func CacheDir() string {
	if d := os.Getenv("QQUESTIO_CACHE_DIR"); d != "" {
		return d
	}
	if home, err := os.UserHomeDir(); err == nil {
		return filepath.Join(home, ".cache", "qquestio")
	}
	// Last-resort fallback to a temp dir so we never silently write to CWD.
	return filepath.Join(os.TempDir(), "qquestio-cache")
}

// safeCollectionName converts a collection name into a filename-safe string.
// Qdrant collection names can contain hyphens, dots, and unicode; this
// normalizes to a portable form.
func safeCollectionName(collection string) string {
	var b strings.Builder
	for _, r := range collection {
		switch {
		case r >= 'A' && r <= 'Z',
			r >= 'a' && r <= 'z',
			r >= '0' && r <= '9',
			r == '-' || r == '_':
			b.WriteRune(r)
		default:
			b.WriteRune('_')
		}
	}
	if b.Len() == 0 {
		return "default"
	}
	return b.String()
}

// CachePath returns the absolute file path used to cache the given collection.
func CachePath(collection string) string {
	return filepath.Join(CacheDir(), safeCollectionName(collection)+".qcache")
}

// LoadCorpusCache loads a cached corpus for the given collection.
// Returns (nil, nil, nil) if the cache file does not exist (a normal case,
// not an error). Returns an error only on actual I/O or format problems.
func LoadCorpusCache(collection string) (*CorpusCache, []QdrantPoint, error) {
	path := CachePath(collection)
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil, nil
		}
		return nil, nil, fmt.Errorf("failed to read cache %s: %w", path, err)
	}
	return parseCacheBytes(collection, data)
}

// parseCacheBytes decodes the binary cache file format.
func parseCacheBytes(collection string, data []byte) (*CorpusCache, []QdrantPoint, error) {
	if len(data) < 4+1+8 {
		return nil, nil, fmt.Errorf("cache file too short (%d bytes)", len(data))
	}
	r := bytes.NewReader(data)

	var magic uint32
	if err := binary.Read(r, binary.LittleEndian, &magic); err != nil {
		return nil, nil, fmt.Errorf("cache: failed to read magic: %w", err)
	}
	if magic != cacheMagic {
		return nil, nil, fmt.Errorf("cache: bad magic 0x%X (expected 0x%X)", magic, cacheMagic)
	}

	var version uint8
	if err := binary.Read(r, binary.LittleEndian, &version); err != nil {
		return nil, nil, fmt.Errorf("cache: failed to read version: %w", err)
	}
	if version != cacheVersion {
		return nil, nil, fmt.Errorf("cache: unsupported version %d", version)
	}

	var tsUnix int64
	if err := binary.Read(r, binary.LittleEndian, &tsUnix); err != nil {
		return nil, nil, fmt.Errorf("cache: failed to read timestamp: %w", err)
	}

	var nameLen uint32
	if err := binary.Read(r, binary.LittleEndian, &nameLen); err != nil {
		return nil, nil, fmt.Errorf("cache: failed to read name length: %w", err)
	}
	nameBytes := make([]byte, nameLen)
	if _, err := r.Read(nameBytes); err != nil {
		return nil, nil, fmt.Errorf("cache: failed to read name: %w", err)
	}
	cachedCollection := string(nameBytes)
	if cachedCollection != collection {
		return nil, nil, fmt.Errorf("cache: collection mismatch (file=%q, requested=%q)", cachedCollection, collection)
	}

	var dim uint32
	if err := binary.Read(r, binary.LittleEndian, &dim); err != nil {
		return nil, nil, fmt.Errorf("cache: failed to read dimension: %w", err)
	}

	var pointCount uint64
	if err := binary.Read(r, binary.LittleEndian, &pointCount); err != nil {
		return nil, nil, fmt.Errorf("cache: failed to read point count: %w", err)
	}

	var payloadLen uint32
	if err := binary.Read(r, binary.LittleEndian, &payloadLen); err != nil {
		return nil, nil, fmt.Errorf("cache: failed to read payload length: %w", err)
	}
	payloadBytes := make([]byte, payloadLen)
	if _, err := r.Read(payloadBytes); err != nil {
		return nil, nil, fmt.Errorf("cache: failed to read payload blob: %w", err)
	}

	var allPayloads []map[string]interface{}
	if err := json.Unmarshal(payloadBytes, &allPayloads); err != nil {
		return nil, nil, fmt.Errorf("cache: failed to decode payload blob: %w", err)
	}
	if uint64(len(allPayloads)) != pointCount {
		return nil, nil, fmt.Errorf("cache: payload count %d != header point count %d", len(allPayloads), pointCount)
	}

	// Read each point: id + vector.
	points := make([]QdrantPoint, 0, pointCount)
	for i := uint64(0); i < pointCount; i++ {
		var idLen uint32
		if err := binary.Read(r, binary.LittleEndian, &idLen); err != nil {
			return nil, nil, fmt.Errorf("cache: failed to read id length for point %d: %w", i, err)
		}
		idBytes := make([]byte, idLen)
		if _, err := r.Read(idBytes); err != nil {
			return nil, nil, fmt.Errorf("cache: failed to read id for point %d: %w", i, err)
		}
		// IDs can be ints, UUIDs, or strings; store as interface{} via JSON round-trip
		// so we don't lose type information.
		var idVal interface{}
		if err := json.Unmarshal(idBytes, &idVal); err != nil {
			return nil, nil, fmt.Errorf("cache: failed to decode id for point %d: %w", i, err)
		}

		var vecLen uint32
		if err := binary.Read(r, binary.LittleEndian, &vecLen); err != nil {
			return nil, nil, fmt.Errorf("cache: failed to read vector length for point %d: %w", i, err)
		}
		expectedVecBytes := uint32(dim) * 4
		if vecLen != expectedVecBytes {
			return nil, nil, fmt.Errorf("cache: vector byte length %d != expected %d (dim=%d) for point %d", vecLen, expectedVecBytes, dim, i)
		}
		vecBytes := make([]byte, vecLen)
		if _, err := r.Read(vecBytes); err != nil {
			return nil, nil, fmt.Errorf("cache: failed to read vector for point %d: %w", i, err)
		}
		// Decode little-endian float32 vector.
		vec := make([]float32, dim)
		vecReader := bytes.NewReader(vecBytes)
		if err := binary.Read(vecReader, binary.LittleEndian, &vec); err != nil {
			return nil, nil, fmt.Errorf("cache: failed to decode vector for point %d: %w", i, err)
		}

		points = append(points, QdrantPoint{
			ID:      idVal,
			Payload: allPayloads[i],
			// Score is meaningless for a cache hit; we will recompute it.
			Score: 0,
		})
	}

	cache := &CorpusCache{
		Collection: cachedCollection,
		Dimension:  int(dim),
		PointCount: int(pointCount),
		CachedAt:   time.Unix(tsUnix, 0),
		filePath:   CachePath(collection),
	}
	return cache, points, nil
}

// SaveCorpusCache persists a corpus to disk. All points must share the same
// vector dimension. Existing cache files for the same collection are replaced.
func SaveCorpusCache(collection string, dim int, points []QdrantPoint) error {
	if len(points) == 0 {
		return fmt.Errorf("refusing to save empty corpus")
	}
	if dim <= 0 {
		return fmt.Errorf("refusing to save with non-positive dimension %d", dim)
	}

	if err := os.MkdirAll(CacheDir(), 0o755); err != nil {
		return fmt.Errorf("failed to create cache dir: %w", err)
	}

	var buf bytes.Buffer

	// Header
	_ = binary.Write(&buf, binary.LittleEndian, cacheMagic)
	_ = binary.Write(&buf, binary.LittleEndian, cacheVersion)
	_ = binary.Write(&buf, binary.LittleEndian, time.Now().Unix())

	nameBytes := []byte(collection)
	_ = binary.Write(&buf, binary.LittleEndian, uint32(len(nameBytes)))
	buf.Write(nameBytes)

	_ = binary.Write(&buf, binary.LittleEndian, uint32(dim))
	_ = binary.Write(&buf, binary.LittleEndian, uint64(len(points)))

	// Payload blob (one JSON array of all payloads)
	allPayloads := make([]map[string]interface{}, len(points))
	for i, p := range points {
		allPayloads[i] = p.Payload
	}
	payloadBytes, err := json.Marshal(allPayloads)
	if err != nil {
		return fmt.Errorf("failed to marshal payloads: %w", err)
	}
	_ = binary.Write(&buf, binary.LittleEndian, uint32(len(payloadBytes)))
	buf.Write(payloadBytes)

	// Each point: id + vector
	for i, p := range points {
		idBytes, err := json.Marshal(p.ID)
		if err != nil {
			return fmt.Errorf("failed to marshal id for point %d: %w", i, err)
		}
		_ = binary.Write(&buf, binary.LittleEndian, uint32(len(idBytes)))
		buf.Write(idBytes)

		if len(p.Vector) != dim {
			return fmt.Errorf("point %d: vector dim %d != expected %d", i, len(p.Vector), dim)
		}
		vecBuf := new(bytes.Buffer)
		if err := binary.Write(vecBuf, binary.LittleEndian, p.Vector); err != nil {
			return fmt.Errorf("failed to encode vector for point %d: %w", i, err)
		}
		vecBytes := vecBuf.Bytes()
		_ = binary.Write(&buf, binary.LittleEndian, uint32(len(vecBytes)))
		buf.Write(vecBytes)
	}

	// Atomic write: write to temp file, then rename.
	path := CachePath(collection)
	tmpPath := path + ".tmp"
	if err := os.WriteFile(tmpPath, buf.Bytes(), 0o644); err != nil {
		return fmt.Errorf("failed to write cache temp file: %w", err)
	}
	if err := os.Rename(tmpPath, path); err != nil {
		_ = os.Remove(tmpPath)
		return fmt.Errorf("failed to rename cache temp file: %w", err)
	}
	return nil
}

// DeleteCorpusCache removes the cache file for the given collection.
// Returns nil if the file does not exist.
func DeleteCorpusCache(collection string) error {
	path := CachePath(collection)
	err := os.Remove(path)
	if err != nil && !os.IsNotExist(err) {
		return err
	}
	return nil
}

// CacheInfo returns a human-readable summary of a cache file (or empty string
// if no cache exists). Used by the `/cache status` slash command.
func CacheInfo(collection string) (string, error) {
	path := CachePath(collection)
	fi, err := os.Stat(path)
	if err != nil {
		if os.IsNotExist(err) {
			return "no cache", nil
		}
		return "", err
	}
	cache, _, err := LoadCorpusCache(collection)
	if err != nil {
		return fmt.Sprintf("cache file present (%s) but failed to parse: %v", formatBytes(fi.Size()), err), nil
	}
	age := time.Since(cache.CachedAt).Truncate(time.Second)
	return fmt.Sprintf("collection=%s dim=%d points=%d size=%s age=%s path=%s",
		cache.Collection, cache.Dimension, cache.PointCount,
		formatBytes(fi.Size()), age, path), nil
}

func formatBytes(n int64) string {
	const (
		KB = 1024
		MB = 1024 * KB
		GB = 1024 * MB
	)
	switch {
	case n >= GB:
		return fmt.Sprintf("%.2f GB", float64(n)/float64(GB))
	case n >= MB:
		return fmt.Sprintf("%.2f MB", float64(n)/float64(MB))
	case n >= KB:
		return fmt.Sprintf("%.2f KB", float64(n)/float64(KB))
	default:
		return fmt.Sprintf("%d B", n)
	}
}
