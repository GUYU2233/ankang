<template>
  <el-card shadow="never">
    <template #header>
      <div class="card-header">
        <span>设备接入（萤石 / 海康 / ONVIF / 模拟）</span>
        <div>
          <el-button size="small" type="primary" @click="load">刷新</el-button>
          <el-button size="small" @click="onSync">同步萤石云设备</el-button>
          <el-button size="small" type="success" @click="openAdd">登记设备</el-button>
        </div>
      </div>
    </template>
    <el-table :data="devices" size="small">
      <el-table-column prop="id" label="ID" width="60" />
      <el-table-column prop="device_name" label="名称" />
      <el-table-column prop="device_serial" label="序列号/地址" width="160" />
      <el-table-column prop="vendor" label="接入方式" width="100" />
      <el-table-column prop="scene" label="场景" width="90" />
      <el-table-column label="归属老人" width="110">
        <template #default="s">
          <el-tag v-if="residentName(s.row)" size="small" type="success">{{ residentName(s.row) }}</el-tag>
          <el-tag v-else size="small" type="info">未绑定</el-tag>
        </template>
      </el-table-column>
      <el-table-column label="视频源" show-overflow-tooltip>
        <template #default="s">{{ sourceText(s.row) }}</template>
      </el-table-column>
      <el-table-column prop="status" label="状态" width="90" />
      <el-table-column label="操作" width="150">
        <template #default="s">
          <el-button size="small" @click="openEdit(s.row)">编辑</el-button>
          <el-button size="small" type="danger" @click="onDelete(s.row)">删除</el-button>
        </template>
      </el-table-column>
    </el-table>

    <el-dialog v-model="dialogVisible" :title="editingId ? '编辑设备' : '登记设备'" width="560px">
      <el-form :model="form" label-width="100px">
        <el-form-item label="设备名称"><el-input v-model="form.device_name" /></el-form-item>
        <el-form-item label="序列号"><el-input v-model="form.device_serial" /></el-form-item>
        <el-form-item label="接入方式">
          <el-select v-model="form.vendor">
            <el-option label="模拟设备" value="sim" />
            <el-option label="萤石云" value="ezviz" />
            <el-option label="海康威视" value="hikvision" />
            <el-option label="ONVIF/RTSP" value="onvif" />
            <el-option label="RTSP直连" value="rtsp" />
          </el-select>
        </el-form-item>
        <el-form-item label="场景">
          <el-select v-model="form.scene">
            <el-option label="客厅" value="客厅" />
            <el-option label="卧室" value="卧室" />
            <el-option label="卫生间" value="卫生间" />
          </el-select>
        </el-form-item>
        <el-form-item label="归属老人">
          <el-select v-model="form.resident_id" clearable filterable placeholder="不绑定老人" style="width:100%">
            <el-option v-for="r in residents" :key="r.id" :label="r.name" :value="r.id" />
          </el-select>
        </el-form-item>
        <el-form-item v-if="form.vendor === 'sim'" label="本地视频">
          <el-select v-model="videoPath" clearable filterable placeholder="留空使用自动合成画面" style="width:100%">
            <el-option v-for="v in videos" :key="v.path" :label="videoLabel(v)" :value="v.path" />
          </el-select>
        </el-form-item>
        <el-form-item label="RTSP/视频地址">
          <el-input v-model="form.access_url" placeholder="rtsp://... 或 本地视频" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="dialogVisible = false">取消</el-button>
        <el-button type="primary" @click="onSave">确定</el-button>
      </template>
    </el-dialog>
  </el-card>
</template>

<script setup>
import { onMounted, reactive, ref, watch } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import api from '../api'

const devices = ref([])
const videos = ref([])
const residents = ref([])
const dialogVisible = ref(false)
const editingId = ref(null)
const videoPath = ref('')
const form = reactive({ device_name: '', device_serial: '', vendor: 'sim', scene: '客厅', access_url: '', model: null, channel_no: 1, enabled: true, resident_id: null })

function sourceText(row) { return row.access_url || '合成画面' }
function videoLabel(v) { return v.scene ? v.name + '（' + v.scene + '）' : v.name }
function residentName(row) {
  if (row.resident_name) return row.resident_name
  const r = residents.value.find(x => x.id === row.resident_id)
  return r ? r.name : ''
}

async function load() { devices.value = await api.devices() }
async function loadVideos() { videos.value = await api.localVideos() }
async function loadResidents() { residents.value = await api.residents() }

function openAdd() {
  editingId.value = null
  videoPath.value = ''
  Object.assign(form, { device_name: '', device_serial: '', vendor: 'sim', scene: '客厅', access_url: '', model: null, channel_no: 1, enabled: true, resident_id: null })
  dialogVisible.value = true
}
function openEdit(row) {
  editingId.value = row.id
  Object.assign(form, {
    device_name: row.device_name, device_serial: row.device_serial,
    vendor: row.vendor, scene: row.scene, access_url: row.access_url || '',
    model: row.model || null, channel_no: row.channel_no || 1,
    enabled: row.enabled !== undefined ? row.enabled : true, resident_id: row.resident_id || null
  })
  const url = row.access_url || ''
  videoPath.value = /.(mp4|avi|mkv)$/i.test(url) ? url : ''
  dialogVisible.value = true
}
watch(videoPath, (v) => { if (form.vendor === 'sim') form.access_url = v || '' })

async function onSave() {
  const payload = { ...form, resident_id: form.resident_id || null }
  if (editingId.value) {
    await api.updateDevice(editingId.value, payload)
    ElMessage.success('设备已更新')
  } else {
    await api.addDevice(payload)
    ElMessage.success('设备已登记')
  }
  dialogVisible.value = false
  load()
}
async function onSync() {
  const r = await api.syncDevices()
  ElMessage.success('同步完成，新增 ' + r.added + ' 台')
  load()
}
async function onDelete(row) {
  await ElMessageBox.confirm('确认删除设备 ' + row.device_name + ' ?', '提示', { type: 'warning' })
  await api.deleteDevice(row.id)
  ElMessage.success('已删除')
  load()
}
onMounted(() => { load(); loadVideos(); loadResidents() })
</script>

<style scoped>
.card-header { display: flex; justify-content: space-between; align-items: center; }
</style>
