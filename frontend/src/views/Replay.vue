<template>
  <div class="replay-page">
    <div class="page-actions">
      <el-button @click="router.back()">返回告警中心</el-button>
      <div>
        <el-tag v-if="alert" :type="isFallEvent ? 'danger' : 'warning'">{{ isFallEvent ? '跌倒警告' : '跌倒风险' }}</el-tag>
        <span class="alert-no">{{ alert?.alert_no || '' }}</span>
      </div>
    </div>

    <el-card shadow="never" class="video-card">
      <template #header>
        <div class="card-header">
          <div>
            <strong>告警录像回放</strong>
            <span class="subtitle">现场记录</span>
          </div>
          <el-tag :type="replayStatus === 'ready' ? 'success' : 'warning'">{{ replayStatusText }}</el-tag>
        </div>
      </template>

      <video v-if="replayStatus === 'ready'" ref="videoRef" :src="replaySrc" controls preload="metadata" playsinline class="player" @loadedmetadata="onLoaded"></video>
      <div v-else-if="['recording', 'encoding'].includes(replayStatus)" class="waiting">
        <el-progress type="circle" :percentage="replayProgress" />
        <div>正在生成告警录像，完成后将自动播放</div>
      </div>
      <el-result v-else-if="replayStatus === 'failed'" icon="error" title="录像生成失败" :sub-title="replayMessage || '请联系管理员检查录像服务'" />
      <el-result v-else icon="warning" title="暂无告警录像" sub-title="此告警未关联历史录像，不能使用实时画面替代核实" />

      <div v-if="replayStatus === 'ready'" class="player-tools">
        <span>当前 {{ formatTime(currentTime) }} / {{ formatTime(duration) }}</span>
        <span>可拖动进度条查看事件经过</span>
        <el-select v-model="playbackRate" size="small" style="width:100px" @change="setRate">
          <el-option label="0.5×" :value="0.5" /><el-option label="1×" :value="1" />
          <el-option label="1.5×" :value="1.5" /><el-option label="2×" :value="2" />
        </el-select>
      </div>
    </el-card>

    <el-card v-if="alert" shadow="never" class="verify-card">
      <template #header><strong>{{ verifyQuestion }}</strong></template>
      <el-descriptions :column="3" border size="small">
        <el-descriptions-item label="告警内容">{{ alert.title }}</el-descriptions-item>
        <el-descriptions-item label="告警等级">{{ alert.level }}</el-descriptions-item>
        <el-descriptions-item label="发生时间">{{ alert.created_at }}</el-descriptions-item>
      </el-descriptions>
      <el-alert title="请完整查看录像后核实。人工结论为最终结果，视觉 AI 仅作低权重辅助。" type="info" :closable="false" style="margin-top:12px" />
      <div v-if="aiResult" class="ai-result">
        AI 辅助判断：<strong>{{ aiResult.decision ? '是' : '否' }}</strong>
        · 置信度 {{ probText(aiResult.confidence) }} · 融合分 {{ probText(aiResult.fused_score) }}
        <span v-if="aiResult.summary"> · {{ aiResult.summary }}</span>
      </div>
      <el-input v-model="verifyNote" type="textarea" :rows="2" placeholder="核验备注（可选）" style="margin:12px 0" />
      <div class="verify-actions">
        <el-button :loading="aiChecking" @click="runAIVerify">AI 辅助确认（权重 0.35）</el-button>
        <el-button type="success" :disabled="replayStatus !== 'ready'" @click="submitVerify(true)">{{ positiveText }}</el-button>
        <el-button type="danger" plain :disabled="replayStatus !== 'ready'" @click="submitVerify(false)">{{ negativeText }}</el-button>
      </div>
    </el-card>
  </div>
</template>

<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import api from '../api'

const route = useRoute(); const router = useRouter()
const alert = ref(null); const videoRef = ref(null)
const replayStatus = ref('recording'); const replayProgress = ref(0); const replayMessage = ref('')
const aiResult = ref(null); const aiChecking = ref(false); const verifyNote = ref('')
const duration = ref(0); const currentTime = ref(0); const playbackRate = ref(1)
let replayTimer = null; let timeTimer = null
const alertId = computed(() => Number(route.params.alertId))
const replaySrc = computed(() => '/api/v1/alerts/' + alertId.value + '/replay/video.mp4')
const isFallEvent = computed(() => alert.value?.event_type === 'fall_event')
const verifyQuestion = computed(() => isFallEvent.value ? '录像中是否确认发生跌倒？' : '录像中是否确认存在跌倒风险？')
const positiveText = computed(() => isFallEvent.value ? '是，确认跌倒' : '是，确认存在风险')
const negativeText = computed(() => isFallEvent.value ? '否，未发生跌倒' : '否，暂未发现风险')
const replayStatusText = computed(() => ({ ready:'录像已就绪', recording:'正在录制', encoding:'正在编码', failed:'生成失败', unavailable:'无录像' }[replayStatus.value] || replayStatus.value))

function probText(v) { return typeof v === 'number' ? (v * 100).toFixed(0) + '%' : '-' }
function formatTime(v) { if (!Number.isFinite(v)) return '00:00'; const s=Math.floor(v); return String(Math.floor(s/60)).padStart(2,'0')+':'+String(s%60).padStart(2,'0') }
function onLoaded() { duration.value = videoRef.value?.duration || 0; setRate() }
function setRate() { if (videoRef.value) videoRef.value.playbackRate = playbackRate.value }
async function refreshReplay() {
  const result = await api.alertReplay(alertId.value)
  replayStatus.value=result.status; replayProgress.value=result.progress||0; replayMessage.value=result.message||''
  if (['ready','failed','unavailable'].includes(result.status) && replayTimer) { clearInterval(replayTimer); replayTimer=null }
}
async function runAIVerify() {
  aiChecking.value=true
  try { aiResult.value=await api.aiVerifyAlert(alertId.value); ElMessage.success('AI 辅助判断已完成') }
  catch(e) { ElMessage.error(e.response?.data?.detail || 'AI 辅助判断失败') }
  finally { aiChecking.value=false }
}
async function submitVerify(decision) {
  try {
    await api.verifyAlert(alertId.value,{operator:'值班员',decision,target:isFallEvent.value?'fall':'risk',note:verifyNote.value})
    ElMessage.success('核验结果已保存'); router.push('/alerts')
  } catch(e) { ElMessage.error(e.response?.data?.detail || '核验失败') }
}
onMounted(async()=>{
  try { alert.value=await api.getAlert(alertId.value) }
  catch(e) { replayStatus.value='failed'; replayMessage.value='告警信息加载失败'; ElMessage.error('告警信息加载失败'); return }
  try { await refreshReplay() }
  catch(e) { replayStatus.value='failed'; replayMessage.value=e.response?.data?.detail || '录像状态加载失败'; ElMessage.error('录像状态加载失败') }
  if (['recording','encoding'].includes(replayStatus.value)) replayTimer=setInterval(refreshReplay,1000)
  timeTimer=setInterval(()=>{ if(videoRef.value) currentTime.value=videoRef.value.currentTime||0 },250)
})
onBeforeUnmount(()=>{ if(replayTimer)clearInterval(replayTimer); if(timeTimer)clearInterval(timeTimer) })
</script>

<style scoped>
.replay-page{max-width:1200px;margin:0 auto}.page-actions,.card-header,.player-tools,.verify-actions{display:flex;align-items:center;justify-content:space-between;gap:12px}.page-actions{margin-bottom:12px}.alert-no,.subtitle{margin-left:10px;color:#909399;font-weight:400}.video-card{background:#fff}.player{display:block;width:100%;max-height:68vh;background:#111;border-radius:6px}.waiting{min-height:480px;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:16px;background:#f5f7fa}.player-tools{padding-top:12px;color:#606266}.verify-card{margin-top:16px}.ai-result{margin-top:12px;padding:10px;background:#f0f7ff;border-radius:4px;color:#337ecc}.verify-actions{justify-content:flex-start;flex-wrap:wrap}@media(max-width:768px){.player{min-height:240px}.waiting{min-height:280px}.player-tools{align-items:flex-start;flex-direction:column}}
</style>
