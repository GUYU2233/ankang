<template>
  <el-row :gutter="16">
    <el-col :span="6">
      <el-card shadow="never">
        <template #header><span>监控通道</span></template>
        <el-menu :default-active="String(currentId)" @select="onSelect">
          <el-menu-item v-for="d in devices" :key="d.id" :index="String(d.id)">
            <span>{{ d.device_name }}（{{ d.scene }}）</span>
            <el-tag size="small" :type="d.status === 'online' ? 'success' : 'info'" style="margin-left:8px;">{{ d.status }}</el-tag>
          </el-menu-item>
        </el-menu>
      </el-card>
    </el-col>
    <el-col :span="18">
      <el-card shadow="never">
        <template #header>
          <div class="card-header">
            <span>实时画面（模拟/测试流）</span>
            <el-button size="small" @click="loadDevices">刷新</el-button>
          </div>
        </template>
        <img v-if="currentId" :key="currentId" :src="streamSrc" style="width:100%;" alt="实时画面" @error="onStreamError" />
        <el-empty v-else description="请选择左侧监控通道" />
        <div style="margin-top:10px;">
          <el-descriptions v-if="meta" :column="4" size="small" border>
            <el-descriptions-item label="状态">{{ stateText }}</el-descriptions-item>
            <el-descriptions-item label="人数">{{ meta.person_count ?? '-' }}</el-descriptions-item>
            <el-descriptions-item label="跌倒">{{ meta.fall_detected ? '是' : '否' }}</el-descriptions-item>
            <el-descriptions-item label="跌倒概率">{{ probText(meta.fall_prob) }}</el-descriptions-item>
            <el-descriptions-item label="风险评分">{{ scoreText(meta.risk_score) }}</el-descriptions-item>
            <el-descriptions-item label="风险等级">{{ levelText }}</el-descriptions-item>
            <el-descriptions-item label="步态不稳">{{ probText(meta.gait_unsteadiness) }}</el-descriptions-item>
            <el-descriptions-item label="跌倒前兆">{{ probText(meta.nearfall_prob) }}</el-descriptions-item>
          </el-descriptions>
        </div>
      </el-card>
    </el-col>
  </el-row>
</template>

<script setup>
import { computed, onMounted, onBeforeUnmount, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import api from '../api'

const router = useRouter()

const route = useRoute()

const devices = ref([])
const currentId = ref(null)
const meta = ref(null)
const pendingAlert = ref(null)
const replayStatus = ref('unavailable')
const replayProgress = ref(0)
const replayTimer = ref(null)
const aiResult = ref(null)
const aiChecking = ref(false)
const verifyNote = ref('')
const streamFailed = ref(false)
const fallbackTick = ref(Date.now())
let metaTimer = null
let fallbackTimer = null

const replaySrc = computed(() => pendingAlert.value ? `/api/v1/alerts/${pendingAlert.value.id}/replay/video.mp4` : '')
const streamSrc = computed(() => streamFailed.value
  ? '/api/v1/streams/' + currentId.value + '/frame.jpg?t=' + fallbackTick.value
  : '/api/v1/streams/' + currentId.value + '/mjpeg')
function onStreamError() {
  streamFailed.value = true
  if (!fallbackTimer) fallbackTimer = setInterval(() => { fallbackTick.value = Date.now() }, 500)
}

const isFallEvent = computed(() => pendingAlert.value?.event_type === 'fall_event')
const verifyQuestion = computed(() => isFallEvent.value ? '现场是否确认发生跌倒？' : '现场是否确认存在跌倒风险？')
const positiveText = computed(() => isFallEvent.value ? '是，确认跌倒' : '是，确认存在风险')
const negativeText = computed(() => isFallEvent.value ? '否，未发生跌倒' : '否，暂未发现风险')
const levelMap = { green: '正常', yellow: '注意', orange: '较高', red: '跌倒/严重' }
const stateText = computed(() => {
  if (!meta.value) return '-'
  if (meta.value.fall_detected) return '跌倒报警'
  return meta.value.source === 'rtsp' ? '直播中' : (meta.value.source === 'local_video' ? '播放中' : meta.value.state || '-')
})
const levelText = computed(() => levelMap[meta.value && meta.value.level] || '-')
function probText(v) { return typeof v === 'number' ? (v * 100).toFixed(0) + '%' : '-' }
function scoreText(v) { return typeof v === 'number' ? (v * 100).toFixed(0) + '%' : '-' }

async function refreshReplay() {
  if (!pendingAlert.value) return
  try {
    const result = await api.alertReplay(pendingAlert.value.id)
    replayStatus.value = result.status
    replayProgress.value = result.progress || 0
    if (['ready', 'failed', 'unavailable'].includes(result.status) && replayTimer.value) {
      clearInterval(replayTimer.value); replayTimer.value = null
    }
  } catch (_) { replayStatus.value = 'failed' }
}
async function loadPendingAlert() {
  const alertId = Number(route.query.alert_id)
  if (!alertId) return
  try {
    pendingAlert.value = await api.getAlert(alertId)
    await refreshReplay()
    if (['recording', 'encoding'].includes(replayStatus.value)) replayTimer.value = setInterval(refreshReplay, 1000)
  } catch (_) { pendingAlert.value = null }
}
async function runAIVerify() {
  if (!pendingAlert.value) return
  aiChecking.value = true
  try { aiResult.value = await api.aiVerifyAlert(pendingAlert.value.id); ElMessage.success('AI 辅助判断已完成') }
  catch (e) { ElMessage.error(e.response?.data?.detail || 'AI 辅助判断失败，请检查视觉巡检配置') }
  finally { aiChecking.value = false }
}
async function submitVerify(decision) {
  if (!pendingAlert.value) return
  try {
    await api.verifyAlert(pendingAlert.value.id, { operator: '值班员', decision, target: isFallEvent.value ? 'fall' : 'risk', note: verifyNote.value })
    ElMessage.success(decision ? (isFallEvent.value ? '已确认跌倒' : '已确认存在风险') : '已记录为误报/暂未发现风险')
    pendingAlert.value = null
    await router.replace({ path: '/live', query: { device_id: currentId.value } })
  } catch (e) { ElMessage.error(e.response?.data?.detail || '核验失败') }
}

async function loadDevices() {
  devices.value = await api.devices()
  const qid = Number(route.query.device_id)
  if (qid && devices.value.some(d => d.id === qid)) {
    currentId.value = qid
  } else if (!currentId.value && devices.value.length) {
    currentId.value = devices.value[0].id
  }
}
function onSelect(index) {
  currentId.value = Number(index)
  streamFailed.value = false
  if (fallbackTimer) { clearInterval(fallbackTimer); fallbackTimer = null }
  meta.value = null
}
async function refreshMeta() {
  if (!currentId.value) return
  try {
    const r = await fetch('/api/v1/streams/' + currentId.value + '/meta')
    const data = await r.json()
    meta.value = data.meta
  } catch (e) { /* ignore */ }
}
onMounted(() => {
  loadDevices()
  loadPendingAlert()
  metaTimer = setInterval(refreshMeta, 800)
  refreshMeta()
})
onBeforeUnmount(() => {
  if (metaTimer) clearInterval(metaTimer)
  if (fallbackTimer) clearInterval(fallbackTimer)
  if (replayTimer.value) clearInterval(replayTimer.value)
})
</script>

<style scoped>
.card-header { display: flex; justify-content: space-between; align-items: center; }
.verify-panel { padding-top: 8px; }
.verify-source { color: #606266; margin-bottom: 6px; }
.ai-result { padding: 8px 10px; background: #f0f7ff; border-radius: 4px; color: #337ecc; }
.replay-wrap { position: relative; background: #111; border-radius: 6px; overflow: hidden; }
.replay-video { display: block; width: 100%; min-height: 360px; max-height: 70vh; background: #111; }
.replay-tag { position: absolute; left: 12px; top: 12px; }
.replay-wait { min-height: 360px; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 12px; background: #f5f7fa; border-radius: 6px; color: #606266; }
</style>
